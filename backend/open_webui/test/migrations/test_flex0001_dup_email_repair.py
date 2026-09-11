"""Tests for the Flexion duplicate-user-email repair migration.

The migration is a restore guard (flexion/flexion-open-webui-infra#580): the
deployed databases are clean, so it is a no-op in normal operation and only
does work if a backup predating the 2026-09-03 hand repair is restored. That
makes it exactly the kind of code that is never exercised until the day it
matters, so it is tested here against a real schema built by the real
migration chain.

Each test creates the tables the repair touches, stamps Alembic at
``a0b1c2d3e4f5`` - the revision this repair is chained to, and the revision
both deployed databases sat at - seeds duplicates, then runs the repair
through the real ``alembic upgrade`` path.

The schema is declared here rather than replayed from the chain because the
pre-v0.11 upstream revisions cannot be applied to an empty SQLite database
with current SQLAlchemy (``018012973d35`` tries to ``DROP INDEX`` a
constraint-backed index). Those revisions were applied incrementally over
years of releases in the environments that matter; replaying them from
scratch is a separate upstream problem and not what this test is about. The
column shapes below are taken from the pre-repair prod copy.
"""

import os

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

REVISION = 'flex0001_dup_email_repair'
BASE_REVISION = 'a0b1c2d3e4f5'

VERSIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'migrations',
    'versions',
)

# A minimal env.py. The real one imports the whole application to get
# DATABASE_URL; here the URL comes straight from the config so the test can
# point at a temporary file.
ENV_PY = '''
from alembic import context
from sqlalchemy import create_engine, pool

connectable = create_engine(
    context.config.get_main_option('sqlalchemy.url'), poolclass=pool.NullPool
)
with connectable.connect() as connection:
    context.configure(connection=connection, target_metadata=None)
    with context.begin_transaction():
        context.run_migrations()
'''


@pytest.fixture
def alembic_cfg(tmp_path):
    """An Alembic config over the real versions directory, on a temp SQLite file."""
    script_dir = tmp_path / 'migrations'
    script_dir.mkdir()
    (script_dir / 'env.py').write_text(ENV_PY)
    os.symlink(VERSIONS_DIR, script_dir / 'versions')

    cfg = Config()
    cfg.set_main_option('script_location', str(script_dir))
    cfg.set_main_option('sqlalchemy.url', f'sqlite:///{tmp_path / "test.db"}')
    return cfg


# Only the tables the repair reads or writes. ``tag`` and ``folder`` carry
# ``user_id`` in their primary key and ``group_member`` / ``access_grant`` in a
# unique constraint - those are the four that can collide when rows are
# repointed, so they matter more than the rest.
SCHEMA = [
    'CREATE TABLE "user" (id VARCHAR(255) NOT NULL PRIMARY KEY, name VARCHAR(255), '
    'email VARCHAR(255), role VARCHAR(255), created_at INTEGER, updated_at INTEGER, '
    'last_active_at INTEGER, oauth JSON)',
    'CREATE TABLE auth (id VARCHAR(255) NOT NULL PRIMARY KEY, email VARCHAR(255), '
    'password TEXT, active INTEGER)',
    'CREATE TABLE chat (id TEXT NOT NULL PRIMARY KEY, user_id TEXT, title TEXT, chat JSON, '
    'share_id TEXT UNIQUE, created_at BIGINT, updated_at BIGINT)',
    'CREATE TABLE chat_message (id TEXT NOT NULL PRIMARY KEY, chat_id TEXT, user_id TEXT, '
    'role TEXT, content TEXT, created_at BIGINT, updated_at BIGINT)',
    'CREATE TABLE tag (id VARCHAR(255) NOT NULL, name VARCHAR(255), user_id VARCHAR(255) NOT NULL, '
    'meta JSON, PRIMARY KEY (id, user_id))',
    'CREATE TABLE folder (id TEXT NOT NULL, user_id TEXT NOT NULL, name TEXT, '
    'PRIMARY KEY (id, user_id))',
    'CREATE TABLE oauth_session (id TEXT NOT NULL PRIMARY KEY, user_id TEXT, provider TEXT, '
    'token TEXT, expires_at BIGINT, created_at BIGINT, updated_at BIGINT)',
    'CREATE TABLE api_key (id TEXT NOT NULL PRIMARY KEY, user_id TEXT, key TEXT UNIQUE)',
    'CREATE TABLE file (id TEXT NOT NULL PRIMARY KEY, user_id TEXT, filename TEXT)',
    'CREATE TABLE group_member (id TEXT NOT NULL PRIMARY KEY, group_id TEXT, user_id TEXT, '
    'UNIQUE (group_id, user_id))',
    'CREATE TABLE access_grant (id TEXT NOT NULL PRIMARY KEY, resource_type TEXT, '
    'resource_id TEXT, principal_type TEXT, principal_id TEXT, permission TEXT, '
    'UNIQUE (resource_type, resource_id, principal_type, principal_id, permission))',
]


@pytest.fixture
def engine(alembic_cfg):
    """A database holding the repair's tables, stamped one revision before it."""
    engine = sa.create_engine(alembic_cfg.get_main_option('sqlalchemy.url'))
    with engine.begin() as conn:
        for ddl in SCHEMA:
            conn.execute(sa.text(ddl))
    command.stamp(alembic_cfg, BASE_REVISION)
    return engine


def _user(conn, user_id, email, last_active_at):
    conn.execute(
        sa.text(
            'INSERT INTO "user" (id, name, email, role, created_at, updated_at, last_active_at) '
            'VALUES (:id, :id, :email, \'user\', 1, 1, :lav)'
        ),
        {'id': user_id, 'email': email, 'lav': last_active_at},
    )
    conn.execute(
        sa.text('INSERT INTO auth (id, email, password, active) VALUES (:id, :email, \'x\', 1)'),
        {'id': user_id, 'email': email or ''},
    )


def _chat(conn, chat_id, user_id):
    conn.execute(
        sa.text(
            'INSERT INTO chat (id, user_id, title, chat, created_at, updated_at) '
            "VALUES (:id, :uid, 't', '{}', 1, 1)"
        ),
        {'id': chat_id, 'uid': user_id},
    )


def _scalar(engine, sql, **params):
    with engine.connect() as conn:
        return conn.execute(sa.text(sql), params).scalar_one()


def test_single_head():
    """The repair must not fork the migration chain.

    This is the guard for the ordering decision described in the module
    docstring of the migration: the repair is chained onto the fork's head so
    that it runs before the upstream v0.11.x revisions. An upstream sync that
    lands new revisions on the same parent would silently produce two heads,
    and ``command.upgrade(cfg, 'head')`` - what the application calls at boot
    - fails outright when there is more than one.
    """
    from alembic.script import ScriptDirectory

    cfg = Config()
    cfg.set_main_option('script_location', os.path.dirname(VERSIONS_DIR))
    heads = ScriptDirectory.from_config(cfg).get_heads()
    assert len(heads) == 1, f'expected exactly one Alembic head, found {heads}'


def test_noop_on_clean_database(alembic_cfg, engine):
    """The common case: no duplicates, nothing changes."""
    with engine.begin() as conn:
        _user(conn, 'a', 'a@example.com', 100)
        _user(conn, 'b', 'b@example.com', 200)
        _chat(conn, 'c1', 'a')

    command.upgrade(alembic_cfg, REVISION)

    assert _scalar(engine, 'SELECT count(*) FROM "user"') == 2
    assert _scalar(engine, 'SELECT count(*) FROM auth') == 2
    assert _scalar(engine, 'SELECT count(*) FROM chat') == 1


def test_merges_onto_most_recently_active_row(alembic_cfg, engine):
    """Keeper is greatest ``last_active_at``, even when the loser is older.

    #580 documents an account whose live row is the *newer* one while the
    abandoned row holds more history, so neither "keep the oldest" nor "keep
    the one with the most chats" is correct.
    """
    with engine.begin() as conn:
        _user(conn, 'keeper', 'dup@example.com', 2000)
        _user(conn, 'loser', 'dup@example.com', 1000)
        _chat(conn, 'keeper-chat', 'keeper')
        # The loser is the older row but holds more history.
        for i in range(5):
            _chat(conn, f'loser-chat-{i}', 'loser')

    command.upgrade(alembic_cfg, REVISION)

    assert _scalar(engine, 'SELECT count(*) FROM "user"') == 1
    assert _scalar(engine, 'SELECT id FROM "user"') == 'keeper'
    assert _scalar(engine, 'SELECT count(*) FROM auth WHERE id = \'loser\'') == 0
    # Nothing deleted: all six chats moved onto the keeper.
    assert _scalar(engine, 'SELECT count(*) FROM chat') == 6
    assert _scalar(engine, "SELECT count(*) FROM chat WHERE user_id = 'keeper'") == 6


def test_case_variant_duplicates_and_three_way_group(alembic_cfg, engine):
    """The index is on ``lower(email)``, so case variants are duplicates too."""
    with engine.begin() as conn:
        _user(conn, 'k', 'Dup@Example.com', 3000)
        _user(conn, 'l1', 'dup@example.com', 2000)
        _user(conn, 'l2', 'DUP@EXAMPLE.COM', 1000)
        _chat(conn, 'c1', 'l1')
        _chat(conn, 'c2', 'l2')

    command.upgrade(alembic_cfg, REVISION)

    assert _scalar(engine, 'SELECT count(*) FROM "user"') == 1
    assert _scalar(engine, "SELECT count(*) FROM chat WHERE user_id = 'k'") == 2


def test_null_email_rows_are_left_alone(alembic_cfg, engine):
    """``uq_user_email_lower`` is partial on ``email IS NOT NULL``.

    Multiple NULL-email rows do not violate it, so the repair must not treat
    them as a duplicate group and merge unrelated accounts together.
    """
    with engine.begin() as conn:
        _user(conn, 'n1', None, 100)
        _user(conn, 'n2', None, 200)

    command.upgrade(alembic_cfg, REVISION)

    assert _scalar(engine, 'SELECT count(*) FROM "user"') == 2


def test_folds_unique_key_collisions(alembic_cfg, engine):
    """A tag both rows own collides on ``tag``'s (id, user_id) primary key.

    This is the only collision the prod repair actually hit - three tags. The
    loser's copy is dropped and the keeper's kept; tags the loser owns alone
    are repointed.
    """
    with engine.begin() as conn:
        _user(conn, 'keeper', 'dup@example.com', 2000)
        _user(conn, 'loser', 'dup@example.com', 1000)
        for tag_id, owner in (('work', 'keeper'), ('work', 'loser'), ('private', 'loser')):
            conn.execute(
                sa.text('INSERT INTO tag (id, name, user_id, meta) VALUES (:id, :id, :uid, NULL)'),
                {'id': tag_id, 'uid': owner},
            )

    command.upgrade(alembic_cfg, REVISION)

    assert _scalar(engine, "SELECT count(*) FROM tag WHERE user_id = 'keeper'") == 2
    assert _scalar(engine, "SELECT count(*) FROM tag WHERE user_id = 'loser'") == 0


def test_drops_losers_oauth_sessions(alembic_cfg, engine):
    """Sessions belong to the row that is going away, so they are not moved.

    Repointing one would hand the keeper a token it never issued.
    """
    with engine.begin() as conn:
        _user(conn, 'keeper', 'dup@example.com', 2000)
        _user(conn, 'loser', 'dup@example.com', 1000)
        for sid, owner in (('s-keep', 'keeper'), ('s-lose', 'loser')):
            conn.execute(
                sa.text(
                    'INSERT INTO oauth_session '
                    '(id, user_id, provider, token, expires_at, created_at, updated_at) '
                    "VALUES (:id, :uid, 'google', 't', 1, 1, 1)"
                ),
                {'id': sid, 'uid': owner},
            )

    command.upgrade(alembic_cfg, REVISION)

    assert _scalar(engine, 'SELECT count(*) FROM oauth_session') == 1
    assert _scalar(engine, 'SELECT id FROM oauth_session') == 's-keep'


def test_tolerates_pre_existing_orphans(alembic_cfg, engine):
    """Orphaned rows already in the database must not trip the assertions.

    The pre-repair prod copy carries 20 of them. The post-conditions check
    that the repair does not *add* orphans, not that there are none.
    """
    with engine.begin() as conn:
        _user(conn, 'keeper', 'dup@example.com', 2000)
        _user(conn, 'loser', 'dup@example.com', 1000)
        _chat(conn, 'orphan', 'user-that-never-existed')

    command.upgrade(alembic_cfg, REVISION)

    assert _scalar(engine, 'SELECT count(*) FROM "user"') == 1
    assert _scalar(engine, "SELECT count(*) FROM chat WHERE user_id = 'user-that-never-existed'") == 1


def test_index_from_f0bd01a18a3d_can_be_built_afterwards(alembic_cfg, engine):
    """The point of the whole exercise.

    ``f0bd01a18a3d`` raises rather than build ``uq_user_email_lower`` over
    duplicate data. After the repair the index must be creatable.
    """
    with engine.begin() as conn:
        _user(conn, 'keeper', 'dup@example.com', 2000)
        _user(conn, 'loser', 'DUP@example.com', 1000)

    command.upgrade(alembic_cfg, REVISION)

    with engine.begin() as conn:
        conn.execute(
            sa.text(
                'CREATE UNIQUE INDEX uq_user_email_lower ON "user" (lower(email)) '
                'WHERE email IS NOT NULL'
            )
        )
