#!/usr/bin/env bash
#
# Sync Flexion's `flex` branch with an upstream open-webui release, by MERGING the
# release tag (never rebasing), and prove afterwards that nothing Flexion owns was
# quietly replaced by upstream's version.
#
#   scripts/upstream-sync.sh merge  <tag>    # on a branch cut from flex: merge + resolve by rule
#   scripts/upstream-sync.sh verify [<tag>]  # after the merge: fail if any Flexion work was lost
#
# The two subcommands are the whole mechanism. CI runs exactly these commands, so a
# failed CI sync is reproduced locally byte-for-byte with the same two lines.
#
# ---------------------------------------------------------------------------
# Why `verify` is built the way it is
# ---------------------------------------------------------------------------
# The historical failure mode on this fork was SILENT: a sync produced a tree in
# which a Flexion change had simply reverted to upstream's version, with no
# conflict, no marker and no error. Two whole features (provider-icon-by-model-id,
# Google OAuth Groups) vanished that way and were caught only by eyeball, twice.
#
# A hand-maintained list of "code that must still be present" cannot catch this:
# it only ever contains the drops someone already noticed. So `verify` derives the
# invariant instead of enumerating it:
#
#   B = merge-base(flex, tag)   -- the upstream release flex was previously on
#   F = flex tip                -- Flexion's work
#   T = the tag being merged
#   M = HEAD                    -- the merge result
#
#   For EVERY path P that Flexion touched (F:P != B:P):
#     * P gone from M            -> a Flexion file was deleted            -> FAIL
#     * M:P == T:P  != F:P       -> Flexion's change reverted to upstream -> FAIL
#
# The second rule is the one that matters: "identical to upstream's version at the
# tag" is exactly what a silent drop looks like, and it is checkable without
# knowing anything about what the change was. Nothing to maintain, no pattern to
# forget. When taking upstream's version IS the right call (lock files; a Flexion
# change upstream has since implemented natively), the path is added to
# .github/upstream-sync-accept-upstream.txt in the same PR -- an explicit, small,
# reviewable diff instead of an invisible revert.
set -euo pipefail

FLEX_REF="${FLEX_REF:-origin/flex}"
# --git-path (not --git-dir) so this works inside a linked worktree, where .git is
# a file rather than a directory.
OUT_DIR="${OUT_DIR:-$(git rev-parse --git-path upstream-sync)}"
ACCEPT_FILE=".github/upstream-sync-accept-upstream.txt"
WF_DIR=".github/workflows"

FAIL=0
fail() { printf '::error::%s\n' "$1" >&2; FAIL=1; }
warn() { printf '::warning::%s\n' "$1" >&2; }
note() { printf '%s\n' "$1" >&2; }

# Blob id of <ref>:<path>, or the empty string when the path does not exist there.
blob() { git rev-parse -q --verify "$1:$2" 2>/dev/null || true; }

log() { printf '| `%s` | %s |\n' "$1" "$2" >> "$OUT_DIR/conflict-resolution-log.md"; }

# Set <path> in the index/worktree to <ref>'s version, deleting it when <ref> has
# no such path. Works on conflicted paths too.
set_from() {
  local ref="$1" path="$2"
  if [ -n "$(blob "$ref" "$path")" ]; then
    git checkout "$ref" -- "$path"
    git add -- "$path"
  else
    git rm -q -f --ignore-unmatch -- "$path"
  fi
}

# Every path under .github/workflows, across flex, the tag and the current index.
workflow_paths() {
  {
    git ls-files -- "$WF_DIR"
    git ls-tree -r --name-only "$FLEX_REF" -- "$WF_DIR"
    [ -n "${1:-}" ] && git ls-tree -r --name-only "$1" -- "$WF_DIR"
  } | sort -u
}

read_accept_list() {
  local ref="${1:-HEAD}"
  if [ -n "$(blob "$ref" "$ACCEPT_FILE")" ]; then
    git show "$ref:$ACCEPT_FILE" | sed -e 's/#.*//' -e 's/[[:space:]]*$//' -e '/^$/d'
  fi
}

# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------
cmd_merge() {
  local target="$1" base f wf
  git rev-parse -q --verify "$target^{commit}" >/dev/null \
    || { fail "Unknown ref: $target. Run: git fetch upstream --tags --prune --force"; exit 1; }
  base=$(git merge-base "$FLEX_REF" "$target") \
    || { fail "No merge base between $FLEX_REF and $target."; exit 1; }

  mkdir -p "$OUT_DIR"
  : > "$OUT_DIR/manual-review.txt"
  {
    printf '# Upstream sync conflict resolution log\n\n'
    printf 'Merging upstream `%s` into `%s` (previous base `%s`).\n\n' \
      "$target" "$FLEX_REF" "$(git describe --tags --abbrev=0 --match 'v[0-9]*' --exclude '*-*' "$base" 2>/dev/null || echo "$base")"
    printf '| File | Resolution |\n|------|------------|\n'
  } > "$OUT_DIR/conflict-resolution-log.md"

  git merge --no-ff --no-commit "$target" || true
  if ! git rev-parse -q --verify MERGE_HEAD >/dev/null; then
    fail "git merge of $target did not start. Already merged into this branch?"
    exit 1
  fi

  local conflicts=()
  while IFS= read -r -d '' f; do conflicts+=("$f"); done < <(git diff -z --name-only --diff-filter=U)

  for f in ${conflicts+"${conflicts[@]}"}; do
    case "$f" in
      "$WF_DIR"/*)
        : # handled wholesale below
        ;;
      functions/*|static/static/providers/*|docs/*|.opencode/*|README_FLEXION.md|*.png|*.ico|*.wasm|*.jpg|*.woff2)
        # Paths upstream has no stake in. Flexion's version wins outright.
        set_from HEAD "$f"
        log "$f" "kept flex's version (Flexion-owned path)"
        ;;
      package-lock.json|*/package-lock.json|uv.lock|*/uv.lock)
        set_from "$target" "$f"
        log "$f" "took upstream's lock file — **regenerate if flex changed dependencies**"
        printf '%s\n' "$f" >> "$OUT_DIR/manual-review.txt"
        ;;
      *)
        # Shared source. Markers stay in the content; a human resolves them and
        # `verify` refuses to pass while any marker survives.
        git add -A -- "$f"
        log "$f" "**conflict — human must resolve** (markers committed)"
        printf '%s\n' "$f" >> "$OUT_DIR/manual-review.txt"
        ;;
    esac
  done

  # ---- .github/workflows is Flexion's, in full -----------------------------
  # flex deliberately renames upstream's workflows to *.disabled so they cannot
  # run on this fork. Taking upstream's versions would undo that AND would make
  # the push introduce new workflow-file content, which the default GITHUB_TOKEN
  # is not permitted to do. Both problems disappear if the merge simply never
  # changes this directory.
  while IFS= read -r wf; do
    [ -n "$wf" ] || continue
    set_from "$FLEX_REF" "$wf"
  done < <(workflow_paths "$target")
  printf '\n`%s/` is kept exactly as it is on `%s`; upstream workflow changes are never taken.\n' \
    "$WF_DIR" "$FLEX_REF" >> "$OUT_DIR/conflict-resolution-log.md"

  if [ -n "$(git diff --name-only --diff-filter=U)" ]; then
    fail "Unresolved paths remain after applying the rules:"
    git diff --name-only --diff-filter=U >&2
    exit 1
  fi

  sort -u -o "$OUT_DIR/manual-review.txt" "$OUT_DIR/manual-review.txt"

  # The manual-review list goes into the commit message, not just into a CI
  # scratch directory. That is what lets `verify` behave identically for a human
  # on a fresh clone and for the CI job that produced the merge.
  {
    printf 'chore: merge upstream %s into flex\n\n' "$target"
    if [ -s "$OUT_DIR/manual-review.txt" ]; then
      printf 'Conflicts in %s file(s) are committed WITH markers and must be resolved\n' \
        "$(wc -l < "$OUT_DIR/manual-review.txt" | tr -d ' ')"
      printf 'before this branch can merge. Run scripts/upstream-sync.sh verify to check.\n\n'
    else
      printf 'No conflicts needed human resolution.\n\n'
    fi
    printf 'Upstream-Sync-Target: %s\n' "$target"
    printf 'Upstream-Sync-Flex-Base: %s\n' "$(git rev-parse "$FLEX_REF")"
    sed 's/^/Upstream-Sync-Manual-Review: /' "$OUT_DIR/manual-review.txt"
  } > "$OUT_DIR/commit-message.txt"

  git commit -q --no-verify -F "$OUT_DIR/commit-message.txt"
  note "Merged $target. $(wc -l < "$OUT_DIR/manual-review.txt" | tr -d ' ') file(s) need human resolution."
}

# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

# Nearest ancestor of HEAD carrying an Upstream-Sync-Target trailer.
find_merge_commit() {
  git rev-list --max-count=50 HEAD | while read -r c; do
    if git log -1 --format=%B "$c" | grep -q '^Upstream-Sync-Target: '; then
      printf '%s\n' "$c"
      return 0
    fi
  done
}

cmd_verify() {
  local target="${1:-}" mc base flex_tip status path m_blob f_blob t_blob b_blob
  local n_checked=0 n_reverted=0 n_missing=0

  mc=$(find_merge_commit || true)
  if [ -z "$mc" ]; then
    fail "No upstream-sync merge commit found in the last 50 commits of HEAD. Run 'merge' first."
    exit 1
  fi
  [ -n "$target" ] || target=$(git log -1 --format=%B "$mc" | sed -n 's/^Upstream-Sync-Target: //p' | head -n1)
  flex_tip=$(git log -1 --format=%B "$mc" | sed -n 's/^Upstream-Sync-Flex-Base: //p' | head -n1)
  [ -n "$flex_tip" ] || flex_tip="$FLEX_REF"
  git rev-parse -q --verify "$flex_tip^{commit}" >/dev/null 2>&1 || flex_tip="$FLEX_REF"

  mkdir -p "$OUT_DIR"
  git log -1 --format=%B "$mc" | sed -n 's/^Upstream-Sync-Manual-Review: //p' | sort -u \
    > "$OUT_DIR/manual-review.txt"

  note "verify: target=$target flex=$flex_tip merge-commit=${mc:0:9}"

  # --- 1. ancestry ---------------------------------------------------------
  # Both sides must genuinely be in HEAD's history. This is what keeps the NEXT
  # sync incremental: upstream's tag is a real ancestor of flex, so merge-base
  # moves forward every release instead of re-resolving years of drift.
  git merge-base --is-ancestor "$flex_tip" HEAD \
    || fail "HEAD does not contain $flex_tip — Flexion commits would be lost."
  git merge-base --is-ancestor "$target" HEAD \
    || fail "HEAD does not contain $target — the release is not really merged, so the next sync will not be incremental."

  # --- 2. no conflict markers, anywhere, ever ------------------------------
  # Unconditional. A list of "files allowed to contain markers" is how markers
  # reach production.
  local marked
  marked=$(git grep -lE -e '^(<<<<<<<|>>>>>>>|=======$)' HEAD -- . 2>/dev/null | sed 's/^HEAD://' || true)
  marked=$(printf '%s\n' "$marked" | sed '/^$/d' | grep -v -x -F -e "$ACCEPT_FILE" || true)
  if [ -n "$marked" ]; then
    printf '%s\n' "$marked" | sed 's/^/  markers: /' >&2
    fail "Conflict markers are still committed in $(printf '%s\n' "$marked" | wc -l | tr -d ' ') file(s). Resolve them, commit, and re-run verify."
  fi

  # --- 3. .github/workflows is untouched by the sync -----------------------
  # Guarantees the push introduces no workflow-file change (so the default
  # GITHUB_TOKEN can push it) and that flex's *.disabled renames survive.
  if ! git diff --quiet "$flex_tip" HEAD -- "$WF_DIR"; then
    git diff --name-status "$flex_tip" HEAD -- "$WF_DIR" | sed 's/^/  workflow: /' >&2
    fail "$WF_DIR differs from $flex_tip. A sync must never change it: upstream's workflows stay disabled on this fork, and GITHUB_TOKEN cannot push workflow changes."
  fi

  # --- 4. the derived no-silent-drop check ---------------------------------
  base=$(git merge-base "$flex_tip" "$target")
  local accepted
  accepted=$(read_accept_list HEAD)

  while IFS=$'\t' read -r status path; do
    [ -n "$path" ] || continue
    case "$path" in "$WF_DIR"/*) continue ;; esac      # covered by check 3
    if printf '%s\n' "$accepted" | grep -q -x -F -- "$path"; then
      note "  accepted-upstream: $path"
      continue
    fi
    n_checked=$((n_checked + 1))

    m_blob=$(blob HEAD "$path")
    f_blob=$(blob "$flex_tip" "$path")
    t_blob=$(blob "$target" "$path")
    b_blob=$(blob "$base" "$path")

    if [ -z "$m_blob" ] && [ -n "$f_blob" ]; then
      note "  MISSING: $path"
      n_missing=$((n_missing + 1))
      continue
    fi
    # The silent-drop signature: the merged file is byte-identical to upstream's
    # version at the tag, while flex's version differed from it.
    if [ -n "$t_blob" ] && [ "$m_blob" = "$t_blob" ] && [ "$f_blob" != "$t_blob" ]; then
      note "  REVERTED TO UPSTREAM: $path"
      n_reverted=$((n_reverted + 1))
      continue
    fi
    # Upstream did not touch it, yet it changed anyway. Legitimate when a human
    # edited it while resolving, so this is a warning, not a failure.
    if [ "$t_blob" = "$b_blob" ] && [ -n "$f_blob" ] && [ "$m_blob" != "$f_blob" ]; then
      warn "$path changed by the sync although upstream ${target} did not touch it — confirm this was intentional."
    fi
  done < <(git diff --no-renames --name-status --diff-filter=AM "$base" "$flex_tip")

  [ "$n_missing" = 0 ] \
    || fail "$n_missing Flexion file(s) are missing from the merge result (listed above)."
  [ "$n_reverted" = 0 ] \
    || fail "$n_reverted Flexion-modified file(s) are byte-identical to upstream ${target} — their Flexion changes were dropped. Restore them, or add the path to $ACCEPT_FILE with a comment explaining why upstream's version is now correct."

  note "verify: checked $n_checked Flexion-touched path(s)."
  if [ "$FAIL" = 0 ]; then
    note "upstream-sync verify: PASS"
  else
    exit 1
  fi
}

case "${1:-}" in
  merge)  cmd_merge  "${2:?usage: $0 merge <upstream-tag>}" ;;
  verify) cmd_verify "${2:-}" ;;
  *) printf 'usage: %s merge <upstream-tag> | verify [<upstream-tag>]\n' "$0" >&2; exit 2 ;;
esac
