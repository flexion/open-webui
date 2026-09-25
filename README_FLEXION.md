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

`flex` takes upstream releases by **merging** the release tag, never by rebasing. Flexion's
commits keep their SHAs, conflicts are resolved once in a single merge commit, and the sync PR
merges normally. Two rules keep this working without any special token:

- **`main` is an exact mirror of upstream.** Never commit to it. GitHub only lets `GITHUB_TOKEN`
  push a workflow file whose exact contents already exist on another branch, and `main` is what
  provides upstream's versions.
- **Workflow files are never hand-merged.** `publish-flex-image.yml` and `upstream-sync.yml` are
  Flexion's; every other file in `.github/workflows/` is kept byte-identical to upstream and
  disabled in the Actions settings instead of renamed.

`scripts/upstream-sync.sh` does the merge and the checks, so the workflow and a local run behave
the same. `verify` fails if the merge lost any of Flexion's commits or files, any pattern in
`.github/upstream-sync-sentinels.txt` (Flexion code inside shared files), or left conflict markers
outside the manual-review list. When you add Flexion code to a shared upstream file, add a
sentinel line for it.

#### Option A — Automated Sync (Recommended)

The **Upstream Sync** workflow runs on the 1st and 15th of each month (09:00 UTC). If upstream has
a `v*.*.*` release newer than the one `flex` is based on, it:

1. Makes sure `main` carries the release, calling GitHub's *Sync fork* API if not. When upstream
   changed workflow files, GitHub refuses that call from `GITHUB_TOKEN`; the run then stops and
   asks a person to click **Sync fork** on `main` and re-run it.
2. Disables any upstream workflow that is active on this fork.
3. Merges the tag into a throwaway `upstream-sync/<tag>-YYYYMMDD-HHMMSS` branch cut from `flex`,
   resolving conflicts by rule (in a merge, `--ours` is `flex`):
   - `functions/`, `static/static/providers/`, `README_FLEXION.md`, binaries → Flexion's version
   - Lock files → upstream's version, flagged for review (regenerate if `flex` changed dependencies)
   - Shared source files → conflict markers committed as-is for a human to resolve in the PR
4. Runs `scripts/upstream-sync.sh verify`, uploads the logs as a run artifact, and opens a PR
   into `flex` (a draft if anything needs manual review), requesting review from `flexion/opencode`.

No secrets are required. Manual runs: **Actions → Upstream Sync → Run workflow**, optionally with a
`target_tag`, or `force: true` to skip the up-to-date and backward-sync checks.

**After the workflow opens a PR:**
1. If files need manual resolution: check out the branch, fix the markers, run
   `scripts/upstream-sync.sh verify <tag>`, push, then mark the PR ready for review
2. Verify Flexion features still work (see checklist in PR body)
3. Merge with **Create a merge commit**. Squash or rebase would drop upstream's history from `flex`
   and break the next sync.
4. Publish the new release to ECR by running the **Publish flex image to ECR** workflow with
   `version=<target_tag> environment=dev` (and `environment=prod` once dev is verified)

---

#### Option B — Manual Sync

Use this when you need direct control, or when the workflow can't finish. Your own push access is
enough.

```bash
# 1. Make sure main mirrors the release (or click "Sync fork" on main in GitHub)
git fetch upstream --tags --force
git fetch origin
git merge-base --is-ancestor vX.Y.Z origin/main || echo "Sync main with upstream first"

# 2. Merge on a throwaway branch
git checkout -b upstream-sync/vX.Y.Z origin/flex
scripts/upstream-sync.sh merge vX.Y.Z

# 3. Resolve the files listed in .git/upstream-sync/manual-review.txt, commit, then check
scripts/upstream-sync.sh verify vX.Y.Z

# 4. Push and open a PR into flex; merge it with "Create a merge commit"
git push -u origin upstream-sync/vX.Y.Z
```

Never force-push `flex`, and never rebase it onto a new release.

---

#### Flexion Customization Inventory

These files contain Flexion-specific changes that must survive every upstream sync:

| File | Purpose | Conflict Risk |
|------|---------|---------------|
| `backend/open_webui/utils/oauth.py` | Google Groups OAuth implementation | High — upstream actively develops auth |
| `backend/open_webui/routers/models.py` | Custom model routing | Medium |
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
