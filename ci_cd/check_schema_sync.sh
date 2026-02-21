#!/usr/bin/env bash
# check_schema_sync.sh
# Ensures all three copies of schema.prisma are identical.
# Exits non-zero if any differ.

set -euo pipefail

SOURCE="schema.prisma"
COPY1="litellm/proxy/schema.prisma"
COPY2="litellm-proxy-extras/litellm_proxy_extras/schema.prisma"

FAILED=0

diff_files() {
    local a="$1"
    local b="$2"
    if ! diff -q "$a" "$b" > /dev/null 2>&1; then
        echo "FAIL: $a and $b are out of sync."
        diff "$a" "$b" || true
        FAILED=1
    else
        echo "OK: $a and $b match."
    fi
}

diff_files "$SOURCE" "$COPY1"
diff_files "$SOURCE" "$COPY2"

if [ "$FAILED" -ne 0 ]; then
    echo ""
    echo "schema.prisma files are out of sync. The source of truth is '$SOURCE'."
    echo "Copy it to the other locations and commit the result."
    exit 1
fi

echo ""
echo "All schema.prisma files are in sync."
