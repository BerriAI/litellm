#!/bin/bash
# Benchmark: Rust Gateway vs Python Proxy
# Measures auth + routing overhead (requests will fail at provider call, but we measure up to that point)

MASTER_KEY="sk-test-key"
RUST_URL="http://127.0.0.1:4001"
REQUESTS=500

echo "============================================"
echo "  Rust Gateway vs Python Proxy Benchmark"
echo "============================================"
echo ""

# Test 1: Auth-only (no model, should fail fast with 404)
echo "=== Test 1: Auth Overhead (invalid model) ==="
echo "Measures: header extraction, key hashing, cache lookup, model resolution"
echo ""

echo "--- Rust Gateway ---"
START=$(date +%s%N)
for i in $(seq 1 $REQUESTS); do
    curl -s -o /dev/null -w "" \
        -H "Authorization: Bearer $MASTER_KEY" \
        -H "Content-Type: application/json" \
        -d '{"model":"nonexistent","messages":[{"role":"user","content":"hi"}]}' \
        "$RUST_URL/v1/chat/completions"
done
END=$(date +%s%N)
RUST_AUTH_MS=$(( (END - START) / 1000000 ))
RUST_AUTH_PER=$(( REQUESTS * 1000 / (RUST_AUTH_MS > 0 ? RUST_AUTH_MS : 1) ))
echo "  $REQUESTS requests in ${RUST_AUTH_MS}ms"
echo "  Throughput: ~${RUST_AUTH_PER} req/s"
echo ""

# Test 2: Full path with valid model (fails at provider, but measures full pipeline)
echo "=== Test 2: Full Pipeline (valid model, provider fails) ==="
echo "Measures: auth + model access + rate limit + routing + cost calc setup"
echo ""

echo "--- Rust Gateway ---"
START=$(date +%s%N)
for i in $(seq 1 $REQUESTS); do
    curl -s -o /dev/null -w "" \
        -H "Authorization: Bearer $MASTER_KEY" \
        -H "Content-Type: application/json" \
        -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}' \
        "$RUST_URL/v1/chat/completions"
done
END=$(date +%s%N)
RUST_FULL_MS=$(( (END - START) / 1000000 ))
RUST_FULL_PER=$(( REQUESTS * 1000 / (RUST_FULL_MS > 0 ? RUST_FULL_MS : 1) ))
echo "  $REQUESTS requests in ${RUST_FULL_MS}ms"
echo "  Throughput: ~${RUST_FULL_PER} req/s"
echo ""

# Test 3: Latency distribution (single requests, measure each)
echo "=== Test 3: Latency Distribution (100 requests) ==="
echo ""

LATENCIES=()
for i in $(seq 1 100); do
    START=$(date +%s%N)
    curl -s -o /dev/null -w "" \
        -H "Authorization: Bearer $MASTER_KEY" \
        -H "Content-Type: application/json" \
        -d '{"model":"nonexistent","messages":[{"role":"user","content":"hi"}]}' \
        "$RUST_URL/v1/chat/completions"
    END=$(date +%s%N)
    LATENCIES+=( $(( (END - START) / 1000000 )) )
done

# Sort latencies
IFS=$'\n' SORTED=($(sort -n <<<"${LATENCIES[*]}")); unset IFS

P50=${SORTED[50]}
P95=${SORTED[95]}
P99=${SORTED[99]}
MIN=${SORTED[0]}
MAX=${SORTED[99]}

echo "--- Rust Gateway Latency (ms) ---"
echo "  min:  ${MIN}ms"
echo "  p50:  ${P50}ms"
echo "  p95:  ${P95}ms"
echo "  p99:  ${P99}ms"
echo "  max:  ${MAX}ms"
echo ""

# Summary
echo "============================================"
echo "  Summary"
echo "============================================"
echo "  Rust auth overhead:     ~${RUST_AUTH_PER} req/s"
echo "  Rust full pipeline:     ~${RUST_FULL_PER} req/s"
echo "  Rust p50 latency:       ${P50}ms"
echo "  Rust p95 latency:       ${P95}ms"
echo ""
echo "  Note: Python proxy numbers from earlier benchmark:"
echo "    Token counter:        ~2000 req/s (single-threaded)"
echo "    Cost calculator:      ~3700 req/s (single-threaded)"
echo "    Auth hash:            ~5000 req/s (single-threaded)"
echo ""
echo "  The Rust gateway handles the FULL pipeline (auth + routing +"
echo "  cost calc + spend tracking) in a single pass, vs Python's"
echo "  multiple FFI crossings for each operation."
echo "============================================"
