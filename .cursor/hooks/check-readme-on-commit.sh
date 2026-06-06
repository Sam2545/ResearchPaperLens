#!/usr/bin/env bash
# Block git commits when staged code changes likely require a README update
# but README.md is not staged. Reads hook JSON from stdin.

set -euo pipefail

input=$(cat)
command=$(INPUT="$input" python3 - <<'PY'
import json
import os

raw = os.environ.get("INPUT", "")
try:
    data = json.loads(raw) if raw.strip() else {}
except json.JSONDecodeError:
    data = {}
print(data.get("command", ""))
PY
)

if ! printf '%s' "$command" | grep -qE 'git[[:space:]]+commit'; then
  printf '%s\n' '{"permission":"allow"}'
  exit 0
fi

staged=$(git diff --cached --name-only 2>/dev/null || true)
if [ -z "$staged" ]; then
  printf '%s\n' '{"permission":"allow"}'
  exit 0
fi

doc_pattern='^(main\.py|requirements\.txt|\.env\.example|src/.+)'
needs_readme=false
has_readme=false

while IFS= read -r file; do
  [ -z "$file" ] && continue
  if [ "$file" = "README.md" ]; then
    has_readme=true
  fi
  if printf '%s\n' "$file" | grep -qE "$doc_pattern"; then
    needs_readme=true
  fi
done <<< "$staged"

if [ "$needs_readme" = false ] || [ "$has_readme" = true ]; then
  printf '%s\n' '{"permission":"allow"}'
  exit 0
fi

cat <<'JSON'
{
  "permission": "deny",
  "user_message": "README.md may be out of date. Update it to reflect your staged changes, stage README.md, and commit again.",
  "agent_message": "Staged changes include main.py, src/, requirements.txt, or .env.example but README.md is not staged. Update README.md to document the new behavior (CLI flags, modules, setup, features, limitations), stage it, and re-run the commit."
}
JSON
