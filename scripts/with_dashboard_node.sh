#!/usr/bin/env bash
set -eu

[ $# -gt 0 ] || { echo "usage: $0 <command> [args...]" >&2; exit 2; }

repo_root=$(cd "$(dirname "$0")/.." && pwd)
dashboard="$repo_root/ui/litellm-dashboard"
floor=$(sed -n 's/.*"node": *">=\([0-9][0-9.]*\)".*/\1/p' "$dashboard/package.json")
pinned=$(tr -d '[:space:]' < "$dashboard/.nvmrc")
floor="${floor:-$pinned}"

meets_floor() {
    awk -v have="$1" -v need="$2" 'BEGIN {
        split(have, h, "."); split(need, n, ".")
        for (i = 1; i <= 3; i++) {
            if (h[i] + 0 < n[i] + 0) exit 1
            if (h[i] + 0 > n[i] + 0) exit 0
        }
    }'
}

current=$(node --version 2>/dev/null | tr -d 'v' || true)
if [ -n "$current" ] && meets_floor "$current" "$floor"; then
    exec "$@"
fi

nvm_script="${NVM_DIR:-$HOME/.nvm}/nvm.sh"
if [ -r "$nvm_script" ]; then
    echo "with_dashboard_node: node ${current:-missing} is below the dashboard floor $floor; switching to $pinned via nvm" >&2
    set +eu
    . "$nvm_script" --no-use || { echo "with_dashboard_node: could not load nvm from $nvm_script" >&2; exit 1; }
    nvm install "$pinned" >&2 || { echo "with_dashboard_node: nvm install $pinned failed" >&2; exit 1; }
    nvm use "$pinned" >&2 || { echo "with_dashboard_node: nvm use $pinned failed" >&2; exit 1; }
    set -eu
    exec "$@"
fi

if command -v fnm > /dev/null 2>&1; then
    echo "with_dashboard_node: node ${current:-missing} is below the dashboard floor $floor; switching to $pinned via fnm" >&2
    fnm install "$pinned" >&2
    eval "$(fnm env)"
    fnm use "$pinned" >&2
    exec "$@"
fi

cat >&2 <<EOF
with_dashboard_node: node ${current:-missing} does not meet ui/litellm-dashboard's engines floor (>= $floor) and neither nvm nor fnm is available to switch automatically.
Fix it with one of:
  - install nvm (https://github.com/nvm-sh/nvm) and re-run; it will pick up node $pinned for you
  - or install/upgrade node yourself to >= $floor (e.g. brew install node), then re-run
EOF
exit 1
