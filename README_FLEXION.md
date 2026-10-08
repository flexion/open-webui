# FlexChat - Flexion's Open WebUI Fork

FlexChat is Flexion's customized deployment of [Open WebUI](https://github.com/open-webui/open-webui), rebranded and configured to integrate with AWS Bedrock via the [Flexion Bedrock Access Gateway](https://github.com/flexion/bedrock-access-gateway).

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `flex` | **Flexion customizations** - Contains all Flexion-specific branding, configurations, and features. This is the primary branch for Flexion development. |
| `main` | Mirrors the upstream Open WebUI `main` branch. Used for tracking upstream releases. |
| `dev` | Mirrors the upstream Open WebUI `dev` branch. Used for tracking upstream development. |

### Keeping Up with Upstream

FlexChat tracks upstream [open-webui/open-webui](https://github.com/open-webui/open-webui). The `flex` branch is rebased onto upstream releases to incorporate new features and fixes while preserving Flexion customizations.

#### One-Time Setup

```bash
# Add upstream remote (if not already configured)
git remote add upstream https://github.com/open-webui/open-webui.git

# Verify remotes
git remote -v
# origin    git@github.com:flexion/open-webui (fetch)
# upstream  https://github.com/open-webui/open-webui.git (fetch)
```

---

#### How syncing works

`flex` takes upstream releases by **merging** the release tag. It is never rebased onto one.

That single choice is the fix for the fork's worst recurring bug. The old rebase-based sync
rewrote Flexion's commits one at a time; in CI `git rebase --continue` failed with no editor
configured and the loop fell through to `git rebase --skip`, which dropped commits without a
word. One sync kept 4 of flex's 17 commits. The rebase also had `--ours`/`--theirs` inverted —
during a rebase `--ours` is the *upstream* side — so every "keep Flexion's version" rule kept
upstream's instead. Two whole features (provider-icon-by-model-id and Google OAuth Groups)
disappeared with no conflict and no error, and were caught only by someone reading the diff.

A merge cannot do that. Flexion's commits keep their SHAs, the release tag becomes a real
ancestor of `flex`, and every conflict is resolved once, in one merge commit, in the open.

Three rules keep it that way:

- **Never squash or rebase a sync PR.** Merge it with *Create a merge commit*. Squashing throws
  away upstream's history, after which `git describe` can no longer tell which release `flex` is
  on and the next sync re-resolves everything from scratch.
- **`flex` owns `.github/workflows/` in full.** Upstream's workflow files live here renamed to
  `*.disabled` so they cannot run on this fork, and a sync never takes upstream's versions of
  them. That keeps the fork from re-activating upstream's release/publish jobs, and it means a
  sync branch introduces no workflow-file change — which is what lets a plain `GITHUB_TOKEN`
  push it.
- **`scripts/upstream-sync.sh` is the whole mechanism.** CI runs exactly the two commands below,
  so any failed CI sync is reproduced locally byte-for-byte.

#### What `verify` actually checks

`scripts/upstream-sync.sh verify` does not look for a list of things that ought to be present —
a list like that only ever contains the drops someone already noticed. It derives the check
instead. For every path Flexion touched relative to the release `flex` was previously on, the
merge result must not be **missing** it and must not be **byte-identical to upstream's version
at the new tag**. "Identical to upstream" is exactly what a silent drop looks like, and it is
checkable without knowing what the change was.

It also refuses any committed conflict marker, anywhere, unconditionally, and refuses any change
to `.github/workflows/`.

When taking upstream's version genuinely *is* right — a lock file, or a Flexion change upstream
has since implemented natively — add the path to `.github/upstream-sync-accept-upstream.txt`
with a comment. That turns the decision into a small reviewable diff instead of an invisible
revert.

`verify` reads the target tag, the `flex` tip and the manual-review list from trailers on the
merge commit, so it needs no arguments and behaves identically in CI and on a fresh clone.

#### Option A — Automated Sync (Recommended)

The **Upstream Sync** workflow runs weekly (Mondays, 09:00 UTC) and can be run from
**Actions → Upstream Sync → Run workflow** with an optional `target_tag`. When upstream has a
`v*.*.*` release newer than the one `flex` is on, it:

1. Cuts `upstream-sync/<tag>-YYYYMMDD-HHMMSS` from `flex` and runs `upstream-sync.sh merge <tag>`,
   which resolves conflicts by rule:
   - `functions/`, `static/static/providers/`, `docs/`, `README_FLEXION.md`, binaries → Flexion's
     version (in a merge, `--ours` really is `flex`)
   - `.github/workflows/` → `flex`'s version, in full
   - lock files → upstream's, flagged for regeneration
   - everything else → conflict markers committed as-is for a human
2. Runs `upstream-sync.sh verify`, uploads the log and the `--remerge-diff` as a run artifact.
3. Pushes the branch and opens a **draft** PR into `flex`.

**The PR is always a draft, and CI never marks it ready.** `verify` proves nothing was silently
dropped; it does not prove the result builds or that the resolutions are semantically right. A
human finishes every sync.

**Finishing a sync:**

```bash
git fetch origin && git fetch upstream --tags --prune --force
git switch upstream-sync/vX.Y.Z-...
# resolve the files listed in the merge commit message, then commit
scripts/upstream-sync.sh verify
npm run build && docker build .
```

Then mark the PR ready. **Upstream Sync Verify** re-runs `verify` on every push to the PR, which
is the only CI this fork has on a `flex`-targeted PR (`backend.yaml` and `frontend.yaml` only
fire on `main`/`dev`). Merge with **Create a merge commit**, then publish to ECR via the
**Publish flex image to ECR** workflow (`version=<tag> environment=dev`, then `prod`).

**Token.** The workflow uses `GITHUB_TOKEN` by default. GitHub refuses a GitHub App token any
push that creates or updates a file under `.github/workflows/`, and there is no `workflows`
entry in a workflow's `permissions:` block to grant. This design sidesteps that by never
changing that directory in a sync. If a push is still rejected (upstream commits that are new to
this remote also carry workflow files), the run says so and offers two fixes: click **Sync fork**
on `main` so those commits already exist here, or add a `SYNC_TOKEN` secret — a GitHub App
installation token, or a fine-grained PAT with *Contents: write*, *Pull requests: write*,
*Workflows: write* — which the workflow uses automatically when present. A classic PAT with the
`workflow` scope also works but needs an org owner to authorize it, which is what stalled the
earlier `UPSTREAM_SYNC_TOKEN` attempt.

---

#### Option B — Manual Sync

Same script, same checks, your own push access. Use it when the workflow cannot finish.

```bash
git fetch origin
git fetch upstream --tags --prune --force

git switch -c upstream-sync/vX.Y.Z origin/flex
scripts/upstream-sync.sh merge vX.Y.Z

# resolve the files the merge commit lists, then commit
scripts/upstream-sync.sh verify

git push -u origin upstream-sync/vX.Y.Z
gh pr create --draft --base flex --head upstream-sync/vX.Y.Z \
  --title "chore: upstream-sync flex onto vX.Y.Z"
```

Never force-push `flex`, and never rebase it onto a release.

---

#### Flexion Customization Inventory

Orientation only. `scripts/upstream-sync.sh verify` derives the real list from the history on
every run, so this table does not need to be complete for a sync to be safe — but keep it
roughly current anyway, because it is what tells a reviewer *what to click through* after a sync.

| File | Purpose | Conflict Risk |
|------|---------|---------------|
| `backend/open_webui/utils/oauth.py` | Google Groups OAuth + admin exemption | High — upstream actively develops auth |
| `backend/open_webui/routers/models.py` | Provider-icon-by-model-id attribution | High — dropped silently twice already |
| `backend/open_webui/migrations/versions/3c9b0ca343fd_*.py` | `down_revision` chained onto `flex0001_dup_email_repair` | High — upstream owns this file; losing the edit gives Alembic two heads and a boot crash loop |
| `Dockerfile` | Node heap bump, IPv6 `no_proxy` strip, `FileResponse` import | High — upstream edits it every release |
| `backend/open_webui/constants.py` | `TASKS.MODEL_RECOMMENDATION` enum value | Low — append-only |
| `backend/open_webui/routers/tasks.py` | `POST /model_recommendation/completions` endpoint | Medium — task routing may change |
| `backend/open_webui/utils/task.py` | `model_recommendation_template()` utility | Low — append-only |
| `src/lib/components/chat/Navbar.svelte` | Flexion navbar changes | Medium |
| `src/lib/components/chat/Placeholder.svelte` | Flexion UI tweak | Low |
| `src/lib/apis/index.ts` | Flexion API additions | Medium |
| `src/lib/components/chat/ModelHelperModal.svelte` | Model selector UI (Flexion-unique) | None — Flexion-only file |
| `functions/` (5 files) | Custom Flexion functions | None — Flexion-only directory |
| `static/static/providers/` (17 files) | Provider icons + metadata | None — Flexion-only directory |
| `README_FLEXION.md` | This file | None — Flexion-only file |
| `docs/oauth-google-groups.md` | OAuth documentation | None — Flexion-only file |

## Local Development Setup

### Prerequisites

- Docker and Docker Compose
- AWS credentials configured (for Bedrock Access Gateway)
- [Flexion Bedrock Access Gateway](https://github.com/flexion/bedrock-access-gateway) running locally (or live connection)

### Architecture Overview

```
┌─────────────────┐     ┌─────────────────────────┐     ┌─────────────────┐
│                 │     │                         │     │                 │
│    FlexChat     │────▶│  Bedrock Access Gateway │────▶│  AWS Bedrock    │
│   (Port 3000)   │     │      (Port 8000)        │     │                 │
│                 │     │                         │     │                 │
└─────────────────┘     └─────────────────────────┘     └─────────────────┘
```

FlexChat connects to the Bedrock Access Gateway (BAG) which provides an OpenAI-compatible API facade for Amazon Bedrock models.

### Step 1: Start the Bedrock Access Gateway

Before running FlexChat, you need the Bedrock Access Gateway running locally. See the [BAG README_FLEXION.md](https://github.com/flexion/bedrock-access-gateway/blob/main/README_FLEXION.md) for detailed setup instructions.

Quick start for BAG:

```bash
# Clone the Bedrock Access Gateway repo
git clone https://github.com/flexion/bedrock-access-gateway.git
cd bedrock-access-gateway

# Set up virtual environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt

# Configure AWS and start the gateway
export AWS_REGION=us-east-1
export API_KEY=bedrock
export ALLOWED_MODEL_IDS='["anthropic.*", "us.anthropic.*", "us.meta.*"]'

# Run on port 8000
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Step 2: Configure FlexChat Environment

Create or update your `.env` file in the FlexChat root directory:

```bash
# Core Settings
ENV=dev
WEBUI_AUTH=FALSE
ENABLE_LOGIN_FORM=false

# Model Configuration
BYPASS_MODEL_ACCESS_CONTROL=true
DEFAULT_MODELS=us.meta.llama3-1-8b-instruct-v1:0

# Bedrock Access Gateway Connection
# Points to the locally running BAG instance
OPENAI_API_BASE_URL=http://host.docker.interal:8000/api/v1
OPENAI_API_KEY=bedrock

# Disable Ollama (we're using Bedrock)
ENABLE_OLLAMA_API=false

# User Settings
DEFAULT_USER_ROLE=user
ENABLE_API_KEYS=true
USER_PERMISSIONS_FEATURES_API_KEYS=true


WEBUI_AUTH=false 
```

### Step 3: Run FlexChat with Docker Compose

```bash
# Build and start FlexChat
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

FlexChat will be available at `http://localhost:3000`.

### Docker Compose Configuration

The `docker-compose.yaml` is configured to:

- Build FlexChat from the local Dockerfile
- Mount a persistent volume for data storage
- Load environment variables from `.env`
- Expose local port services like the BAG
- Add `host.docker.internal` for accessing host services (like the BAG)

```yaml
services:
  open-webui:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: open-webui
    volumes:
      - open-webui:/app/backend/data
    network_mode: 'host'
    env_file:
      - .env
    extra_hosts:
      - host.docker.internal:host-gateway
    restart: unless-stopped
```

**Note:** The Ollama service is included in the compose file but is not required when using Bedrock. You can remove or comment out the Ollama service and its dependency if desired.

## Connecting to Bedrock Access Gateway

The `OPENAI_API_BASE_URL` environment variable is set to `http://host.docker.internal:8000/api/v1` to connect FlexChat to the locally running Bedrock Access Gateway.

### Important Notes

- **API Key:** The `OPENAI_API_KEY=bedrock` matches the default API key used by the BAG in local development mode.
- **Available Models:** The models available in FlexChat depend on the `ALLOWED_MODEL_IDS` configured in the Bedrock Access Gateway.

### Updating OPENAI_API_BASE_URL for Docker

If FlexChat is running in Docker and BAG is running on your host:

```bash
# In .env, use host.docker.internal for Docker-to-host communication
OPENAI_API_BASE_URL=http://host.docker.internal:8000/api/v1
```

## Flexion Customizations

The `flex` branch includes the following Flexion-specific changes:

### Branding
- Application name changed from "Open WebUI" to "FlexChat"
- Custom Flexion logo used for favicons and splash screens
- Updated site manifest and HTML title

### Configuration Defaults
- Default integration with Bedrock Access Gateway
- Ollama disabled by default
- Google OAuth pre-configured (credentials required)

## Troubleshooting

### FlexChat can't connect to models

1. Verify the Bedrock Access Gateway is running on port 8000
2. Check that `OPENAI_API_BASE_URL` is correctly set
3. If using Docker, try using `0.0.0.0` and setting `network_mode: 'host'`

### No models appearing in the UI

1. Check BAG logs for any authentication errors
2. Verify your AWS credentials have Bedrock invoke permissions
3. Confirm `ALLOWED_MODEL_IDS` in BAG includes the models you expect

### Docker build fails

1. Ensure Docker has sufficient resources allocated
2. Try clearing Docker cache: `docker-compose build --no-cache`

## Related Documentation

- [Open WebUI Documentation](https://docs.openwebui.com/)
- [Flexion Bedrock Access Gateway](https://github.com/flexion/bedrock-access-gateway/blob/main/README_FLEXION.md)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
