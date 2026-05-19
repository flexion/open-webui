# Session: FlexChat v0.8.10 → v0.9.5 Upstream Sync + Production Deployment

**Branch**: flex  
**Issue**: N/A (maintenance / upstream sync)  
**Created**: 2026-05-18  
**Status**: in-progress (production stabilization ongoing)

---

## Goal

1. Rebase the `flex` branch from upstream open-webui v0.8.10 → v0.9.5 (801 commits)
2. Build and deploy the new image to production (`chat.cloud.flexion.us`)
3. Document the process and create a GitHub Actions pipeline for future automated syncs

---

## Approach

- **Rebase strategy** (not merge) — matches the established `feat: rebase Flexion customizations onto vX.Y.Z` pattern
- **Throwaway branch** — all rebase work on `flex-rebase-onto-v0.9.5`, never directly on `flex`
- **HITL gate** — draft PR for human review before `flex` is updated
- **CI pipeline** — `workflow_dispatch` only, Bedrock Claude for AI conflict resolution, draft PR output

---

## Session Log

- **2026-05-18**: Full upstream sync session — rebase, production deployment, incident response

---

## Key Decisions

### Rebase Strategy
- Rebase (not merge) onto `upstream/main` to maintain clean linear history
- Throwaway branch `flex-rebase-onto-v0.9.5` used — never rebase `flex` directly
- Backup tag `flex-backup-pre-rebase-20260518` created and pushed to origin before any rebase work

### CI Pipeline Design
- `workflow_dispatch` only (manual trigger) — human always approves before `flex` is updated
- Conflict resolution hierarchy: binary → `--ours` | lock files → `--theirs` | Flexion-unique → `--ours` | shared source → Bedrock Claude
- LLM capped at 10 files; clean rebases still get a draft PR
- OIDC federation for AWS credentials (no static keys)
- `SYNC_PAT` required (not `GITHUB_TOKEN`) — needed to push branches containing `.github/workflows/`

### OAUTH_ALLOWED_ROLES Decision
- v0.9.5 strictly enforces `OAUTH_ALLOWED_ROLES` — users not in any matching group get `ACCESS_PROHIBITED`
- v0.8.10 silently fell through to `DEFAULT_USER_ROLE=user` when no match found
- No single all-staff Google group exists at Flexion that all employees are in
- **Decision**: Set `ENABLE_OAUTH_ROLE_MANAGEMENT=false` — rely on `OAUTH_ALLOWED_DOMAINS` (flexion.us) for access control instead of group membership

### WEBUI_SECRET_KEY
- Currently regenerated on every task start (16 bytes — below recommended 32)
- Causes `InvalidToken` errors on OAuth sessions from previous deployments
- **TODO**: Pin to a fixed Secrets Manager secret to survive redeploys

---

## Bugs Found During Testing

### Bug 1: `FileResponse` not imported in `models.py`
- **File**: `backend/open_webui/routers/models.py` line 35
- **Symptom**: 500 on every model profile image request (`/api/v1/models/model/profile/image`)
- **Root cause**: `FileResponse` used at line 599 but never imported — `NameError` → 500
- **Fix**: Added `FileResponse` to `from fastapi.responses import ...`
- **Commit**: `29299e736`

### Bug 2: `model_recommendation_template` not async
- **Files**: `backend/open_webui/utils/task.py`, `backend/open_webui/routers/tasks.py`
- **Symptom**: `TypeError: Object of type coroutine is not JSON serializable` on model recommendation endpoint
- **Root cause**: `prompt_template()` became `async` in v0.9.5; Flexion's `model_recommendation_template` still called it synchronously
- **Fix**: Made `model_recommendation_template` async, added `await` at call site
- **Commit**: `e156747d1`

### Bug 3: `STATIC_DIR` not imported in `models.py`
- **File**: `backend/open_webui/routers/models.py` line 40
- **Symptom**: `NameError: name 'STATIC_DIR' is not defined` on startup → container crash
- **Root cause**: `STATIC_DIR` is no longer a global in v0.9.5 — needs explicit import from `open_webui.config`
- **Fix**: Added `STATIC_DIR` to `from open_webui.config import BYPASS_ADMIN_ACCESS_CONTROL, STATIC_DIR`
- **Commit**: `e156747d1`

### Bug 4: Dockerfile build failure (OrbStack `no_proxy` IPv6 CIDR)
- **File**: `Dockerfile` line 141
- **Symptom**: Build fails with `httpx.InvalidURL: Invalid port: 'b51a:cc66:f0::'` during model download step
- **Root cause**: OrbStack injects IPv6 CIDR ranges (e.g. `fd07:b51a:cc66::/64`) into `no_proxy`; `httpx` cannot parse CIDR notation as URL patterns
- **Fix**: Strip IPv6 CIDR entries from `no_proxy`/`NO_PROXY` at the top of the failing `RUN` block
- **Commit**: `29299e736`

---

## Production Deployment Incidents (2026-05-18)

### Incident 1: CDK deploy dropped all OAuth config (rev 27)
- **Cause**: `yarn cdk:deploy:aws:prod` run without sourcing `.env.prod` — all env vars (`GOOGLE_CLIENT_ID`, `OPENAI_API_KEY`, etc.) were absent, so CDK deployed with empty config
- **Symptom**: Google Sign In button disappeared from login page
- **Resolution**: Rolled back to rev 26 via `aws ecs update-service --task-definition :26`, then redeployed with `source .env.prod && yarn cdk:deploy:aws:prod`
- **Prevention**: `deploy:aws:prod` script now includes `--context useEcr=true`; `.env.prod` file created (gitignored) with all required vars

### Incident 2: `ResourceInitializationError: ecr:GetAuthorizationToken denied`
- **Cause**: CDK deploy without `--context useEcr=true` skips the ECR permission block in the execution role inline policy, removing `ecr:GetAuthorizationToken`
- **Resolution**: Manually attached `arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy` to the execution role; added `--context useEcr=true` permanently to `deploy:aws:prod` script
- **PR**: flexion/flexion-open-webui-infra#458

### Incident 3: `EMAIL_TAKEN` error on Google login for existing users
- **Cause**: v0.9.5 moved `oauth_sub` from a dedicated text column to a JSON `oauth` column. Existing users have their sub in the old column; `get_user_by_oauth_sub()` returns null → falls through to signup → finds email exists → throws `EMAIL_TAKEN`
- **Resolution**: Set `OAUTH_MERGE_ACCOUNTS_BY_EMAIL=true` — falls back to email lookup and migrates sub to new column on first login
- **PR**: flexion/flexion-open-webui-infra#458

### Incident 4: UI takes 90+ seconds to load after login
- **Cause**: Gemini pipe function (`functions/pipes/gemini.pipe.py`) reads `GOOGLE_API_KEY` but task definition only had `GOOGLE_GEMINI_API_KEY`. Empty key → Google auth timeout → 30s hang per function × 3 functions = 90s
- **Resolution**: Added `GOOGLE_API_KEY` as a second alias pointing to the same Secrets Manager secret as `GOOGLE_GEMINI_API_KEY`
- **PR**: flexion/flexion-open-webui-infra#458

### Incident 5: `ACCESS_PROHIBITED` for non-admin users
- **Cause**: v0.9.5 strictly enforces `OAUTH_ALLOWED_ROLES` — users not in a matching group are denied. `OAUTH_ALLOWED_ROLES` was never set in any historical task definition (v0.8.10 silently fell through). `all-employees@flexion.us` does not exist as a Google group.
- **Resolution**: Set `ENABLE_OAUTH_ROLE_MANAGEMENT=false` — rely on `OAUTH_ALLOWED_DOMAINS` for access control
- **PR**: flexion/flexion-open-webui-infra#458

### Incident 6: lgarceau@flexion.us lost chat history (duplicate user accounts)
- **Cause**: Multiple new user IDs created for `lgarceau@flexion.us` during the deployment chaos window (before `OAUTH_MERGE_ACCOUNTS_BY_EMAIL` was set). Chats belong to original user `03a4a27f`; currently logged in as `e2cd097e`
- **Status**: **UNRESOLVED** — requires DB migration to reassign chats from old user ID to current
- **Other users**: Unaffected — no one else successfully logged in during the broken window

---

## Files Changed (open-webui repo, `flex` branch)

| File | Change |
|------|--------|
| `backend/open_webui/routers/models.py` | Added `FileResponse` + `STATIC_DIR` imports |
| `backend/open_webui/utils/task.py` | Made `model_recommendation_template` async |
| `backend/open_webui/routers/tasks.py` | Added `await` to `model_recommendation_template` call |
| `Dockerfile` | Strip IPv6 CIDR from `no_proxy` before model download step |
| `README_FLEXION.md` | Expanded "Keeping Up with Upstream" from stub to full runbook |
| `.github/workflows/upstream-sync.yml` | New: GitHub Actions workflow with Bedrock AI conflict resolution |

## Files Changed (flexion-open-webui-infra repo, PR #458)

| File | Change |
|------|--------|
| `cdk-infra/lib/constructs/open-webui-service.ts` | Added `OAUTH_MERGE_ACCOUNTS_BY_EMAIL`, `GOOGLE_API_KEY`, `OAUTH_ALLOWED_ROLES`/`ENABLE_OAUTH_ROLE_MANAGEMENT` props |
| `cdk-infra/lib/cdk-infra-stack.ts` | Wired new props |
| `cdk-infra/bin/cdk-infra.ts` | Read new props from env vars |
| `cdk-infra/package.json` | Added `--context useEcr=true` to `deploy:aws:prod` script |
| `.gitignore` | Added `.env.prod`, `.env.dev`, `.env.local`, `.env` |

---

## Upstream Sync Process (Documented)

### One-time setup
```bash
git remote add upstream https://github.com/open-webui/open-webui.git
```

### Manual sync runbook (see README_FLEXION.md for full detail)
```bash
# 1. Fetch and create safety net
git fetch upstream && git fetch origin
git tag flex-backup-pre-rebase-$(date +%Y%m%d) flex
git push origin flex-backup-pre-rebase-$(date +%Y%m%d)

# 2. Throwaway branch
git checkout -b flex-rebase-onto-vX.Y.Z flex

# 3. Rebase
git rebase upstream/main

# 4. Resolve conflicts (binary→--ours, lock→--theirs, flexion-unique→--ours, shared→manual)

# 5. Verify
git merge-base --is-ancestor upstream/main flex-rebase-onto-vX.Y.Z && echo PASS
git log --oneline flex-rebase-onto-vX.Y.Z ^upstream/main | wc -l  # expect N Flexion commits

# 6. Push + draft PR
git push --force-with-lease origin flex-rebase-onto-vX.Y.Z
gh pr create --draft --base flex --head flex-rebase-onto-vX.Y.Z

# 7. After review: update flex
git checkout flex
git push --force-with-lease origin flex-rebase-onto-vX.Y.Z:flex

# 8. Update origin/main
git checkout main && git merge --ff-only upstream/main && git push origin main

# 9. Cleanup
git branch -d flex-rebase-onto-vX.Y.Z
git push origin --delete flex-rebase-onto-vX.Y.Z
```

### Automated CI sync
- Workflow: `.github/workflows/upstream-sync.yml`
- Trigger: Actions → Upstream Sync → Run workflow
- Default: `dry_run: true` (safe to check drift without side effects)
- Required secrets: `SYNC_PAT`, `AWS_BEDROCK_ROLE_ARN`, `AWS_REGION`

### Deploy after sync
```bash
# 1. Build and push image
cd open-webui
aws ecr get-login-password --region us-east-2 | \
  docker login --username AWS --password-stdin 380270640373.dkr.ecr.us-east-2.amazonaws.com
docker build -t 380270640373.dkr.ecr.us-east-2.amazonaws.com/open-webui-prod:latest .
docker push 380270640373.dkr.ecr.us-east-2.amazonaws.com/open-webui-prod:latest

# 2. Deploy CDK (always source .env.prod first)
cd flexion-open-webui-infra
source .env.prod && yarn cdk:deploy:aws:prod
```

---

## Key Takeaways for Future Syncs

### v0.9.5 Breaking Changes (Flexion-specific)
1. **`prompt_template()` is now async** — any Flexion utility that calls it must be `async` and use `await`
2. **`STATIC_DIR` is no longer a global** — must be explicitly imported from `open_webui.config`
3. **`OAUTH_ALLOWED_ROLES` is strictly enforced** — must be set or `ENABLE_OAUTH_ROLE_MANAGEMENT=false`
4. **`oauth_sub` moved to JSON column** — `OAUTH_MERGE_ACCOUNTS_BY_EMAIL=true` required for existing users
5. **`FileResponse` must be explicitly imported** — not re-exported from fastapi.responses automatically

### CDK Deploy Checklist (ALWAYS do before deploying)
```bash
# Must source .env.prod — CDK reads ALL config from env vars at synth time
source .env.prod && yarn cdk:deploy:aws:prod
```

### Things to Fix Before Next Sync
- [ ] Pin `WEBUI_SECRET_KEY` to a fixed Secrets Manager secret (currently regenerated each deploy → invalidates all sessions)
- [ ] Fix `lgarceau@flexion.us` duplicate user accounts (DB migration needed)
- [ ] Rotate Gemini API key (`AIzaSyAGSYHw7CIrizlaOEN0Mx0E8QLaEZup-5Y` is invalid)
- [ ] Enable ECS Exec on the service for future DB debugging
- [ ] Tag ECR images with version + date (not just `latest`) for easier rollback

### Flexion Customization Inventory (conflict-risk files)
| File | Purpose | v0.9.5 Risk |
|------|---------|-------------|
| `backend/open_webui/utils/oauth.py` | Google Groups OAuth | High — auth infrastructure changes frequently |
| `backend/open_webui/routers/models.py` | Provider icons + model routing | Medium |
| `backend/open_webui/constants.py` | `TASKS.MODEL_RECOMMENDATION` enum | Low |
| `backend/open_webui/routers/tasks.py` | Model recommendation endpoint | Medium — task routing changed |
| `backend/open_webui/utils/task.py` | `model_recommendation_template()` | Low — but must stay async |
| `src/lib/components/chat/Navbar.svelte` | Flexion navbar | Medium |
| `src/lib/components/chat/Placeholder.svelte` | Flexion UI tweak | Low |
| `src/lib/apis/index.ts` | Flexion API additions | Medium |
| `functions/` (5 files) | Custom Flexion functions | None — Flexion-only |
| `static/static/providers/` (17 files) | Provider icons | None — Flexion-only |

---

## Next Steps

- [ ] Fix `ENABLE_OAUTH_ROLE_MANAGEMENT=false` — redeploy infra (PR #458)
- [ ] Fix `lgarceau@flexion.us` duplicate user — DB migration via one-off ECS task
- [ ] Pin `WEBUI_SECRET_KEY` in Secrets Manager
- [ ] Rotate Gemini API key
- [ ] Merge PR #458 once approved
- [ ] Kill the one-off diagnostic ECS task (`46cab7e6`)
