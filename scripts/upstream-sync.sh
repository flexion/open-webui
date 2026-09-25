#!/usr/bin/env bash
# Merge an upstream release tag into a branch cut from flex, then check that no Flexion work was lost.
#
#   scripts/upstream-sync.sh merge  <tag>   # on the sync branch: merge, resolve by rule, commit
#   scripts/upstream-sync.sh verify <tag>   # after merge: exit non-zero if anything Flexion-owned is gone
#   scripts/upstream-sync.sh is-flex-workflow <path>
#
# FLEX_REF (default origin/flex) is the flex tip the branch was cut from. MIRROR_REF (default
# origin/main) is the branch that mirrors upstream. Logs go to OUT_DIR (default inside .git).
set -eo pipefail

FLEX_REF="${FLEX_REF:-origin/flex}"
MIRROR_REF="${MIRROR_REF:-origin/main}"
OUT_DIR="${OUT_DIR:-$(git rev-parse --git-dir)/upstream-sync}"
SENTINELS=".github/upstream-sync-sentinels.txt"
# Workflows Flexion owns. Every other workflow file is kept identical to upstream's.
FLEX_WORKFLOWS=".github/workflows/publish-flex-image.yml .github/workflows/upstream-sync.yml"

FAIL=0
bad() { echo "::error::$1"; FAIL=1; }

is_flex_workflow() {
  local w
  for w in $FLEX_WORKFLOWS; do [ "$1" = "$w" ] && return 0; done
  return 1
}

log() { printf '| `%s` | %s |\n' "$1" "$2" >> "$OUT_DIR/conflict-resolution-log.md"; }

# Resolve a conflicted path to one side. In a merge, stage 2 is flex and stage 3 is upstream.
take() {
  local stage="$1" path="$2" side
  [ "$stage" = 2 ] && side=--ours || side=--theirs
  if git ls-files -u -- "$path" | awk '{print $3}' | grep -qx "$stage"; then
    git checkout "$side" -- "$path"
    git add -- "$path"
  else
    git rm -q -f -- "$path"
  fi
}

set_from() {
  local ref="$1" path="$2"
  if git cat-file -e "$ref:$path" 2>/dev/null; then
    git checkout "$ref" -- "$path"
  else
    git rm -q -f --ignore-unmatch -- "$path"
  fi
}

cmd_merge() {
  local target="$1" f wf
  mkdir -p "$OUT_DIR"
  printf '# Upstream sync conflict resolution log\n\nMerging upstream `%s` into `%s`.\n\n| File | Resolution |\n|------|------------|\n' \
    "$target" "$FLEX_REF" > "$OUT_DIR/conflict-resolution-log.md"
  : > "$OUT_DIR/manual-review.txt"

  git merge --no-ff --no-commit "$target" || true
  if ! git rev-parse -q --verify MERGE_HEAD >/dev/null; then
    echo "::error::git merge of ${target} did not start. Is it already merged into this branch?"
    exit 1
  fi

  local conflicts=()
  while IFS= read -r -d '' f; do conflicts+=("$f"); done < <(git diff -z --name-only --diff-filter=U)

  for f in "${conflicts[@]}"; do
    case "$f" in
      .github/workflows/*)
        ;; # normalized below
      functions/*|static/static/providers/*|README_FLEXION.md|*.png|*.ico|*.wasm)
        take 2 "$f"
        log "$f" "kept flex's version"
        ;;
      package-lock.json|*/package-lock.json|uv.lock|*/uv.lock)
        take 3 "$f"
        log "$f" "took upstream's lockfile; **regenerate it if flex changed dependencies**"
        echo "$f" >> "$OUT_DIR/manual-review.txt"
        ;;
      *)
        git add -A -- "$f"
        log "$f" "**left for manual review** (conflict markers committed)"
        echo "$f" >> "$OUT_DIR/manual-review.txt"
        ;;
    esac
  done

  # Flexion-owned workflows come from flex, everything else from upstream, byte for byte.
  # That keeps every workflow blob equal to a branch tip, which lets GITHUB_TOKEN push it.
  while IFS= read -r wf; do
    if is_flex_workflow "$wf"; then set_from "$FLEX_REF" "$wf"; else set_from "$target" "$wf"; fi
  done < <({ git ls-files -- .github/workflows
             git ls-tree -r --name-only "$target" -- .github/workflows
             git ls-tree -r --name-only "$FLEX_REF" -- .github/workflows; } | sort -u)
  printf '\nWorkflow files: %s kept from flex; all others set to upstream'"'"'s version.\n' \
    "$FLEX_WORKFLOWS" >> "$OUT_DIR/conflict-resolution-log.md"

  if [ -n "$(git diff --name-only --diff-filter=U)" ]; then
    echo "::error::Unresolved paths remain after applying the rules:"
    git diff --name-only --diff-filter=U
    exit 1
  fi

  sort -u -o "$OUT_DIR/manual-review.txt" "$OUT_DIR/manual-review.txt"
  git commit -q -m "chore: merge upstream ${target} into flex"
}

cmd_verify() {
  local target="$1" base missing wf blob path pattern content marked unexpected
  mkdir -p "$OUT_DIR"
  touch "$OUT_DIR/manual-review.txt"

  git merge-base --is-ancestor "$FLEX_REF" HEAD || bad "HEAD does not contain ${FLEX_REF}; flex commits would be lost."
  git merge-base --is-ancestor "$target" HEAD || bad "HEAD does not contain ${target}."
  git merge-base --is-ancestor "$target" "$MIRROR_REF" \
    || bad "${target} is not on ${MIRROR_REF}; pushing would add upstream commits GITHUB_TOKEN may not be allowed to push. Sync the fork first."

  # Files Flexion added on top of upstream must all survive the merge.
  base=$(git merge-base "$FLEX_REF" "$target")
  missing=$(comm -23 \
    <(git diff --no-renames --name-only --diff-filter=A "$base" "$FLEX_REF" | grep -v '^\.github/workflows/' | sort) \
    <(git ls-tree -r --name-only HEAD | sort))
  if [ -n "$missing" ]; then
    printf '%s\n' "$missing" | sed 's/^/  missing: /'
    bad "$(printf '%s\n' "$missing" | wc -l | tr -d ' ') Flexion file(s) are missing after the merge."
  fi

  for wf in $FLEX_WORKFLOWS; do
    [ "$(git rev-parse -q --verify "HEAD:$wf" || true)" = "$(git rev-parse -q --verify "$FLEX_REF:$wf" || true)" ] \
      || bad "${wf} differs from ${FLEX_REF}."
  done

  # GitHub rejects a GITHUB_TOKEN push that adds a workflow file unless the same path and
  # contents already exist on another branch, so each one must match flex or the mirror.
  while IFS= read -r wf; do
    blob=$(git rev-parse "HEAD:$wf")
    [ "$blob" = "$(git rev-parse -q --verify "$FLEX_REF:$wf" || true)" ] \
      || [ "$blob" = "$(git rev-parse -q --verify "$MIRROR_REF:$wf" || true)" ] \
      || bad "${wf} matches neither ${FLEX_REF} nor ${MIRROR_REF}, so GITHUB_TOKEN cannot push it."
  done < <(git ls-tree -r --name-only HEAD -- .github/workflows)

  # Sentinels catch Flexion code inside shared files, which the file check above can't see.
  if git cat-file -e "HEAD:$SENTINELS" 2>/dev/null; then
    while IFS='|' read -r path pattern; do
      case "$path" in ''|'#'*) continue ;; esac
      if [ "$path" = '*' ]; then
        git grep -qE -e "$pattern" HEAD -- . || bad "Sentinel not found anywhere in the tree: ${pattern}"
      else
        content=$(git show "HEAD:$path" 2>/dev/null) && grep -qE -e "$pattern" <<<"$content" \
          || bad "Sentinel not found in ${path}: ${pattern}"
      fi
    done < <(git show "HEAD:$SENTINELS")
  else
    echo "::warning::${SENTINELS} not found; skipping sentinel checks."
  fi

  # Conflict markers are only allowed in files explicitly queued for manual review.
  marked=$(git grep -lE -e '^(<<<<<<<|>>>>>>>)( |$)' HEAD -- . | sed 's/^HEAD://' | sort -u || true)
  unexpected=$(comm -23 <(printf '%s\n' "$marked" | sed '/^$/d') <(sort -u "$OUT_DIR/manual-review.txt"))
  if [ -n "$unexpected" ]; then
    printf '%s\n' "$unexpected" | sed 's/^/  markers: /'
    bad "Conflict markers found in files not queued for manual review."
  fi

  [ "$FAIL" = 0 ] || exit 1
  echo "upstream-sync verify: all checks passed"
}

case "${1:-}" in
  merge) cmd_merge "${2:?usage: $0 merge <upstream-tag>}" ;;
  verify) cmd_verify "${2:?usage: $0 verify <upstream-tag>}" ;;
  is-flex-workflow) is_flex_workflow "${2:?usage: $0 is-flex-workflow <path>}" ;;
  *) echo "usage: $0 merge|verify <upstream-tag> | is-flex-workflow <path>" >&2; exit 2 ;;
esac
