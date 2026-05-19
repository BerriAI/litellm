#!/usr/bin/env bash
# Concurrent TPM bypass test — mints a virtual key with tpm_limit=100
# (api_key scope in the v3 rate-limiter), races 10 concurrent calls,
# prints a verdict, then deletes the key.
#
# Note: the `tpm: 100` on a model_list deployment is the *router's*
# load-balancing TPM, not a v3 rate-limit descriptor. The v3 limiter
# enforces against limits set on the key/team/user — so we set
# tpm_limit=100 on the key itself.
#
# Pre-PR: ~all 10 return 200 (race lets concurrent requests bypass the limit).
# Post-PR: only ~1–2 fit under tpm_limit=100, rest return 429.
#
# Setup (separate terminal):
#   kubectl port-forward -n litellm svc/yassin-veks-litellm-helm 4000:4000
#
# Run:
#   bash scripts/tpm_headline_test.sh
set -u
PROXY="${PROXY:-http://localhost:4000}"
MASTER_KEY="${MASTER_KEY:-sk-perf-test-fixed-do-not-rotate}"
MODEL="${MODEL:-opus-4.6}"

echo "=== Concurrent TPM bypass test ==="
echo "proxy=$PROXY  model=$MODEL  key tpm_limit=100  concurrency=10  max_tokens=50"
echo

gen_resp=$(curl -s -X POST "$PROXY/key/generate" \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"models\":[\"$MODEL\"],\"tpm_limit\":100,\"duration\":\"10m\",\"key_alias\":\"tpm-headline-$$-$(date +%s)\"}")

KEY=$(printf '%s' "$gen_resp" | sed -n 's/.*"key"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
if [ -z "$KEY" ]; then
  echo "FAIL — could not mint virtual key. Response: $gen_resp"
  exit 1
fi
echo "Minted virtual key: ${KEY:0:12}…"
echo

cleanup() {
  curl -s -X POST "$PROXY/key/delete" \
    -H "Authorization: Bearer $MASTER_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"keys\":[\"$KEY\"]}" > /dev/null 2>&1 || true
  [ -n "${tmp:-}" ] && rm -rf "$tmp"
}
trap cleanup EXIT

tmp=$(mktemp -d)
for i in $(seq 1 10); do
  ( curl -s -o "$tmp/body.$i" -w "%{http_code}" \
      "$PROXY/v1/chat/completions" \
      -H "Authorization: Bearer $KEY" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"concurrent tpm test $i\"}],\"max_tokens\":50}" \
      > "$tmp/code.$i" ) &
done
wait

ok=0; limited=0; other=0
for i in $(seq 1 10); do
  code=$(cat "$tmp/code.$i")
  case "$code" in
    200) ok=$((ok+1)) ;;
    429) limited=$((limited+1)) ;;
    *)   other=$((other+1)); echo "req $i -> $code: $(cat "$tmp/body.$i" | head -c 200)" ;;
  esac
done

echo
echo "Results: 200=$ok  429=$limited  other=$other"
if [ "$limited" -ge 1 ] && [ "$ok" -ge 1 ]; then
  echo "PASS — reservation enforced under concurrency."
  exit 0
elif [ "$ok" -eq 10 ]; then
  echo "FAIL — all 10 succeeded; concurrent bypass still possible."
  exit 1
else
  echo "INCONCLUSIVE — investigate non-200/429 above."
  exit 2
fi
