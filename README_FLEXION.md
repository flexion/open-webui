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

#### Option A — Automated Sync (Recommended)

Use the **Upstream Sync** GitHub Actions workflow for future syncs. It handles drift detection, rebase, AI-assisted conflict resolution via Amazon Bedrock, and opens a draft PR for your review.

**Prerequisites — configure these secrets once in repo Settings → Secrets and variables → Actions:**

| Secret | Description |
|--------|-------------|
| `SYNC_PAT` | GitHub Personal Access Token with `repo` + `workflow` scopes. Required because `GITHUB_TOKEN` cannot push branches that contain `.github/workflows/` files. |
| `AWS_BEDROCK_ROLE_ARN` | IAM role ARN with `bedrock:InvokeModel` permission on `anthropic.claude-3-5-sonnet-20241022-v2:0`. Trust policy must allow `token.actions.githubusercontent.com` for `repo:flexion/open-webui:ref:refs/heads/*`. |
| `AWS_REGION` | AWS region where Bedrock is available (e.g., `us-east-1`). |

**Triggering the workflow:**

1. Go to **Actions → Upstream Sync → Run workflow**
2. Set `dry_run: false` (default is `true` — safe to run first to check drift)
3. Leave `target_ref` as default (`refs/heads/main`) to sync to upstream's latest release
4. Click **Run workflow**

**What the workflow does:**
1. Detects how many commits `flex` is behind upstream
2. Creates a throwaway branch `upstream-sync/YYYYMMDD-HHMMSS` from `flex`
3. Rebases onto upstream, resolving conflicts automatically:
   - Binary files (`*.png`, `*.ico`, `*.wasm`) → keeps Flexion's version (`--ours`)
   - Lock files (`package-lock.json`, `uv.lock`) → takes upstream's version (`--theirs`)
   - Flexion-unique files (`functions/`, `static/static/providers/`, `README_FLEXION.md`) → keeps Flexion's version
   - Shared source files → Amazon Bedrock Claude resolves (capped at 10 files; beyond that, raw markers left for manual review)
4. Pushes the throwaway branch and opens a **draft PR** targeting `flex`
5. The draft PR includes a conflict resolution log and HITL review checklist

**After the workflow opens a draft PR:**
1. Review the conflict resolution log in the PR description
2. Verify Flexion features still work (see checklist in PR body)
3. Approve and merge the draft PR
4. Then fast-forward `flex` locally:
   ```bash
   git checkout flex
   git pull origin flex
   ```

---

#### Option B — Manual Rebase Runbook

Use this when you need direct control, or when the automated workflow encounters issues.

**Step 1 — Safety prep**

```bash
# Fetch latest from both remotes
git fetch upstream
git fetch origin

# Create a backup tag (recovery point)
git tag flex-backup-pre-rebase-$(date +%Y%m%d) flex
git push origin flex-backup-pre-rebase-$(date +%Y%m%d)

# Create a throwaway working branch (never rebase flex directly)
git checkout -b flex-rebase-onto-vX.Y.Z flex
```

**Step 2 — Rebase**

```bash
git rebase upstream/main
```

**Step 3 — Resolve conflicts** (if any)

Use this priority order for each conflicted file:

| File Type | Command | Rationale |
|-----------|---------|-----------|
| Binary (`*.png`, `*.ico`, `*.wasm`) | `git checkout --ours <file> && git add <file>` | Not text-mergeable; Flexion icons are custom |
| Lock files (`package-lock.json`, `uv.lock`) | `git checkout --theirs <file> && git add <file>` | Regenerated deterministically; take upstream's |
| Flexion-unique (`functions/`, `static/static/providers/`, `README_FLEXION.md`) | `git checkout --ours <file> && git add <file>` | Entirely Flexion additions; upstream never touches these |
| Shared source files (`oauth.py`, `models.py`, etc.) | Manual merge | Preserve Flexion intent, incorporate upstream structure |

After resolving each file: `git add <file>` then `git rebase --continue`

If a commit becomes empty after resolution: `git rebase --skip`

If the rebase becomes unresolvable: `git rebase --abort` (your throwaway branch returns to its pre-rebase state)

**Step 4 — Verify**

```bash
# Confirm upstream/main is an ancestor of the rebased branch
git merge-base --is-ancestor upstream/main flex-rebase-onto-vX.Y.Z && echo "PASS"

# Confirm Flexion commits are on top (should be 3)
git log --oneline flex-rebase-onto-vX.Y.Z ^upstream/main

# Confirm no merge commits (clean linear history)
git log --merges flex-rebase-onto-vX.Y.Z ^upstream/main | wc -l  # must be 0
```

**Step 5 — Push and open draft PR**

```bash
git push --force-with-lease origin flex-rebase-onto-vX.Y.Z

gh pr create \
  --draft \
  --base flex \
  --head flex-rebase-onto-vX.Y.Z \
  --title "feat: rebase Flexion customizations onto vX.Y.Z" \
  --body "Upstream sync: vPREV → vX.Y.Z. See conflict log for details."
```

**Step 6 — After human review and approval**

```bash
# Fast-forward flex to the rebased branch
git checkout flex
git merge --ff-only flex-rebase-onto-vX.Y.Z
git push --force-with-lease origin flex

# Update origin/main to mirror upstream/main
git checkout main
git merge --ff-only upstream/main
git push origin main

# Clean up throwaway branch
git branch -d flex-rebase-onto-vX.Y.Z
git push origin --delete flex-rebase-onto-vX.Y.Z
```

Commit message pattern: `feat: rebase Flexion customizations onto vX.Y.Z`

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
