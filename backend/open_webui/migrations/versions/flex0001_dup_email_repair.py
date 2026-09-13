"""repair duplicate user emails (Flexion restore guard)

Revision ID: flex0001_dup_email_repair
Revises: a0b1c2d3e4f5
Create Date: 2026-09-10

Fork-owned data repair. Merges duplicate ``user`` rows that share a
case-folded email onto a single keeper row, so that upstream revision
``f0bd01a18a3d_add_unique_normalized_user_email_index`` can build
``uq_user_email_lower``.

Background: flexion/flexion-open-webui-infra#580. A May 2026 storage
incident briefly presented an empty database to the application; OAuth
users were treated as new and re-created as second rows. Both deployed
databases were repaired by hand on 2026-09-03 and are clean today, so
this revision is a **restore guard**, not an upgrade blocker: restoring
any backup that predates the hand repair resurrects the duplicates and
re-blocks the upgrade, and nothing else would repair them a second time.

Design, mirroring the ``merge.py`` used on prod (quoted verbatim in #580):

* Keeper is the row with the greatest ``last_active_at``. Neither "oldest
  row" nor "most chats" is correct - #580 documents an account whose live
  row is the newer one while the abandoned row holds more history.
* Every other row's content is repointed onto the keeper across every
  table carrying a user reference, then that row and its ``auth`` record
  are deleted. Merge, not quarantine: prod received a merge and it
  preserves history that quarantining would strand.
* ``chat`` and ``chat_message`` rows are never deleted. This is enforced
  structurally, not by convention - see ``NEVER_DELETE_FROM``.
* Rows that cannot be repointed because they would collide with a row the
  keeper already owns (for example a ``tag`` both accounts created) are
  folded: the loser's copy is dropped and the keeper's is kept.
* The whole repair runs inside Alembic's transaction. Post-conditions are
  asserted before returning; any failure raises, which rolls the whole
  thing back rather than committing a partial repair.

On an already-clean database this is a no-op: one grouped SELECT that
returns no rows, and an immediate return.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = 'flex0001_dup_email_repair'
down_revision: str | None = 'a0b1c2d3e4f5'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

log = logging.getLogger('alembic.runtime.migration')

# Content in these tables is never deleted, only repointed. If a row here
# could not be repointed the migration aborts rather than folding it.
NEVER_DELETE_FROM = frozenset({'chat', 'chat_message'})

# Per-login rows belonging to a row that is going away. Repointing a dead
# session onto the keeper would resurrect a token the keeper never issued,
# so these are dropped instead of moved (the same choice merge.py made).
DROP_NOT_MOVE = frozenset({'oauth_session'})

# Tables that reference a user through something other than a ``user_id``
# column: table -> (column holding the user id, extra equality predicate).
EXTRA_USER_REFS: dict[str, tuple[str, dict[str, str]]] = {
    'access_grant': ('principal_id', {'principal_type': 'user'}),
}


def _quote(name: str) -> str:
    return op.get_bind().dialect.identifier_preparer.quote(name)


def _duplicate_groups(conn: sa.Connection) -> list[tuple[str, list[str]]]:
    """Return [(lower_email, [user_id, ...]), ...] for emails held by >1 row.

    Ordered so the keeper is first: greatest ``last_active_at``, then
    ``updated_at`` / ``created_at`` as tie-breakers, then ``id`` so the
    choice is deterministic even for rows that are identical on all three.
    """
    rows = conn.execute(
        sa.text(
            """
            SELECT lower(email) AS email_key,
                   id,
                   COALESCE(last_active_at, 0) AS lav,
                   COALESCE(updated_at, 0) AS upd,
                   COALESCE(created_at, 0) AS crt
            FROM "user"
            WHERE email IS NOT NULL
              AND lower(email) IN (
                  SELECT lower(email)
                  FROM "user"
                  WHERE email IS NOT NULL
                  GROUP BY lower(email)
                  HAVING count(*) > 1
              )
            """
        )
    ).fetchall()

    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(row.email_key, []).append(row)

    result = []
    for email_key in sorted(grouped):
        members = sorted(
            grouped[email_key],
            key=lambda r: (r.lav, r.upd, r.crt, r.id),
            reverse=True,
        )
        result.append((email_key, [m.id for m in members]))
    return result


def _user_ref_tables(inspector: sa.Inspector) -> list[tuple[str, str, dict[str, str]]]:
    """Discover every table holding a user reference.

    Returns (table, user_column, extra_predicate). Discovery is reflective
    rather than hardcoded because the set of tables carrying ``user_id``
    grows with almost every upstream release.
    """
    found = []
    for table in sorted(inspector.get_table_names()):
        if table in {'user', 'auth', 'alembic_version'}:
            continue
        columns = {c['name'] for c in inspector.get_columns(table)}
        if 'user_id' in columns:
            found.append((table, 'user_id', {}))
            continue
        extra = EXTRA_USER_REFS.get(table)
        if extra and extra[0] in columns and set(extra[1]) <= columns:
            found.append((table, extra[0], dict(extra[1])))
    return found


def _unique_key_sets(inspector: sa.Inspector, table: str, user_column: str) -> list[list[str]]:
    """Unique constraints on ``table`` that include ``user_column``.

    Only these can collide when rows are repointed onto the keeper. A
    constraint that does not mention the user column (``chat.id``,
    ``api_key.key``) is unaffected by a change of owner.
    """
    candidates: list[list[str]] = []
    pk = inspector.get_pk_constraint(table).get('constrained_columns') or []
    if pk:
        candidates.append(list(pk))
    for uq in inspector.get_unique_constraints(table):
        candidates.append(list(uq['column_names']))
    for ix in inspector.get_indexes(table):
        if ix.get('unique') and all(c is not None for c in ix['column_names']):
            candidates.append([c for c in ix['column_names']])

    key_sets = []
    seen = set()
    for cols in candidates:
        if user_column not in cols:
            continue
        others = [c for c in cols if c != user_column]
        if not others:
            # The user column alone is unique - the loser row simply cannot
            # coexist. Nothing sensible to fold; treat as a collision below.
            others = []
        marker = tuple(sorted(others))
        if marker in seen:
            continue
        seen.add(marker)
        key_sets.append(others)
    return key_sets


def _where(column_values: dict[str, object], prefix: str) -> tuple[str, dict]:
    """Build a portable equality WHERE fragment, NULL-safe without dialect tricks."""
    clauses, params = [], {}
    for i, (col, value) in enumerate(column_values.items()):
        if value is None:
            clauses.append(f'{_quote(col)} IS NULL')
        else:
            key = f'{prefix}{i}'
            clauses.append(f'{_quote(col)} = :{key}')
            params[key] = value
    return ' AND '.join(clauses) if clauses else '1 = 1', params


def _fold_collisions(
    conn: sa.Connection,
    table: str,
    user_column: str,
    extra: dict[str, str],
    keeper: str,
    loser: str,
    key_sets: list[list[str]],
) -> int:
    """Drop the loser's rows that would violate a unique key once repointed."""
    folded = 0
    qt = _quote(table)
    for key_cols in key_sets:
        base_where, base_params = _where(dict(extra), 'x')

        select_cols = ', '.join(_quote(c) for c in key_cols) if key_cols else '1'
        keeper_keys = {
            tuple(r) if key_cols else ()
            for r in conn.execute(
                sa.text(
                    f'SELECT {select_cols} FROM {qt} '
                    f'WHERE {_quote(user_column)} = :owner AND {base_where}'
                ),
                {'owner': keeper, **base_params},
            ).fetchall()
        }
        if not keeper_keys:
            continue

        loser_rows = conn.execute(
            sa.text(
                f'SELECT {select_cols} FROM {qt} '
                f'WHERE {_quote(user_column)} = :owner AND {base_where}'
            ),
            {'owner': loser, **base_params},
        ).fetchall()

        for row in loser_rows:
            key = tuple(row) if key_cols else ()
            if key not in keeper_keys:
                continue
            if table in NEVER_DELETE_FROM:
                raise RuntimeError(
                    f'Refusing to repair duplicate users: repointing {table} rows from '
                    f'{loser} onto {keeper} would collide on {key_cols or [user_column]}, '
                    f'and rows in {table} are never deleted. Resolve this by hand.'
                )
            key_where, key_params = _where(dict(zip(key_cols, key)), 'k')
            deleted = conn.execute(
                sa.text(
                    f'DELETE FROM {qt} WHERE {_quote(user_column)} = :owner '
                    f'AND {base_where} AND {key_where}'
                ),
                {'owner': loser, **base_params, **key_params},
            ).rowcount
            folded += deleted
            log.info(
                'dup-email-repair: folded %s row(s) in %s for loser %s (collides on %s=%s)',
                deleted,
                table,
                loser,
                key_cols or [user_column],
                key,
            )
    return folded


def _orphan_counts(
    conn: sa.Connection, ref_tables: list[tuple[str, str, dict[str, str]]]
) -> dict[str, int]:
    """Count rows pointing at a user id that does not exist.

    Measured before and after, never asserted to be zero: the deployed
    databases already carry pre-existing orphans (dead ``access_grant``
    principals, ``chat_message`` rows from users deleted long ago). The
    repair must not add to them, but it is not this revision's job to
    clean them up.
    """
    counts = {}
    for table, user_column, extra in ref_tables:
        where, params = _where(dict(extra), 'x')
        counts[table] = conn.execute(
            sa.text(
                f'SELECT count(*) FROM {_quote(table)} '
                f'WHERE {where} AND {_quote(user_column)} IS NOT NULL '
                f'AND {_quote(user_column)} NOT IN (SELECT id FROM "user")'
            ),
            params,
        ).scalar_one()
    counts['auth'] = conn.execute(
        sa.text('SELECT count(*) FROM auth WHERE id NOT IN (SELECT id FROM "user")')
    ).scalar_one()
    return counts


def upgrade() -> None:
    if context.is_offline_mode():
        # A data repair cannot be expressed as static SQL: it depends on the
        # rows present. Emitting nothing is correct and leaves f0bd01a18a3d
        # to complain if the operator's data actually has duplicates.
        log.warning(
            'dup-email-repair: offline mode, skipping duplicate user email repair. '
            'Run migrations online against a database that has duplicates.'
        )
        return

    conn = op.get_bind()
    groups = _duplicate_groups(conn)
    if not groups:
        # The overwhelmingly common case: both deployed databases are clean.
        return

    inspector = sa.inspect(conn)
    ref_tables = _user_ref_tables(inspector)
    key_sets = {
        table: _unique_key_sets(inspector, table, user_column)
        for table, user_column, _ in ref_tables
    }

    before_users = conn.execute(sa.text('SELECT count(*) FROM "user"')).scalar_one()
    before_chats = conn.execute(sa.text('SELECT count(*) FROM chat')).scalar_one()
    before_messages = conn.execute(sa.text('SELECT count(*) FROM chat_message')).scalar_one()
    before_orphans = _orphan_counts(conn, ref_tables)
    losers_total = sum(len(ids) - 1 for _, ids in groups)

    log.warning(
        'dup-email-repair: %s duplicate email group(s) covering %s row(s); '
        'merging %s loser row(s) onto their keeper across %s table(s).',
        len(groups),
        sum(len(ids) for _, ids in groups),
        losers_total,
        len(ref_tables),
    )

    moved = folded = dropped_sessions = 0

    for email_key, member_ids in groups:
        keeper, losers = member_ids[0], member_ids[1:]
        log.warning(
            'dup-email-repair: %s -> keeping %s, merging %s',
            email_key,
            keeper,
            ', '.join(losers),
        )
        for loser in losers:
            for table, user_column, extra in ref_tables:
                qt = _quote(table)
                base_where, base_params = _where(dict(extra), 'x')

                if table in DROP_NOT_MOVE:
                    dropped = conn.execute(
                        sa.text(
                            f'DELETE FROM {qt} WHERE {_quote(user_column)} = :owner '
                            f'AND {base_where}'
                        ),
                        {'owner': loser, **base_params},
                    ).rowcount
                    dropped_sessions += dropped
                    continue

                folded += _fold_collisions(
                    conn, table, user_column, extra, keeper, loser, key_sets[table]
                )

                updated = conn.execute(
                    sa.text(
                        f'UPDATE {qt} SET {_quote(user_column)} = :keeper '
                        f'WHERE {_quote(user_column)} = :owner AND {base_where}'
                    ),
                    {'keeper': keeper, 'owner': loser, **base_params},
                ).rowcount
                moved += updated
                if updated:
                    log.info(
                        'dup-email-repair: repointed %s row(s) in %s from %s to %s',
                        updated,
                        table,
                        loser,
                        keeper,
                    )

            conn.execute(sa.text('DELETE FROM auth WHERE id = :id'), {'id': loser})
            conn.execute(sa.text('DELETE FROM "user" WHERE id = :id'), {'id': loser})

    _assert_repaired(
        conn, before_users, before_chats, before_messages, losers_total, ref_tables, before_orphans
    )

    log.warning(
        'dup-email-repair: done. %s row(s) repointed, %s folded on unique-key collisions, '
        '%s stale oauth_session row(s) dropped, %s user+auth row(s) removed. '
        'user: %s -> %s. chat and chat_message untouched (%s / %s).',
        moved,
        folded,
        dropped_sessions,
        losers_total,
        before_users,
        before_users - losers_total,
        before_chats,
        before_messages,
    )


def _assert_repaired(
    conn: sa.Connection,
    before_users: int,
    before_chats: int,
    before_messages: int,
    losers_total: int,
    ref_tables: list[tuple[str, str, dict[str, str]]],
    before_orphans: dict[str, int],
) -> None:
    """Post-conditions. Raising here rolls the whole repair back."""
    problems = []

    remaining = conn.execute(
        sa.text(
            'SELECT count(*) FROM (SELECT lower(email) FROM "user" WHERE email IS NOT NULL '
            'GROUP BY lower(email) HAVING count(*) > 1) AS dups'
        )
    ).scalar_one()
    if remaining:
        problems.append(f'{remaining} duplicate email group(s) still present')

    after_users = conn.execute(sa.text('SELECT count(*) FROM "user"')).scalar_one()
    if after_users != before_users - losers_total:
        problems.append(f'user count {after_users}, expected {before_users - losers_total}')

    after_chats = conn.execute(sa.text('SELECT count(*) FROM chat')).scalar_one()
    if after_chats != before_chats:
        problems.append(f'chat count changed: {before_chats} -> {after_chats}')

    after_messages = conn.execute(sa.text('SELECT count(*) FROM chat_message')).scalar_one()
    if after_messages != before_messages:
        problems.append(f'chat_message count changed: {before_messages} -> {after_messages}')

    for table, after in _orphan_counts(conn, ref_tables).items():
        before = before_orphans.get(table, 0)
        if after > before:
            problems.append(f'{table} gained {after - before} orphaned row(s) ({before} -> {after})')

    if problems:
        raise RuntimeError(
            'Duplicate user email repair failed its post-conditions and has been rolled '
            'back; nothing was written. Problems: ' + '; '.join(problems)
        )


def downgrade() -> None:
    """Irreversible by construction.

    The merge discards the loser rows' identity, so there is nothing to
    restore. Downgrading past this revision is a no-op rather than an
    error, so that an operator stepping the chain backwards for an
    unrelated reason is not blocked.
    """
    log.warning(
        'dup-email-repair: downgrade is a no-op - merged user rows cannot be un-merged. '
        'Restore from a backup if you need the pre-merge state.'
    )
