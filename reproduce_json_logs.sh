#!/bin/bash

# Script to reproduce json_logs issue with actual litellm proxy server
# This will start the proxy, trigger errors, and show log output

set -e

echo "================================================================================"
echo "LITELLM JSON LOGS REPRODUCER - Real Proxy Server Test"
echo "================================================================================"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ ! -z "$PROXY_PID" ]; then
        echo "Stopping litellm proxy (PID: $PROXY_PID)..."
        kill $PROXY_PID 2>/dev/null || true
        wait $PROXY_PID 2>/dev/null || true
    fi
    rm -f proxy_output.log
}

trap cleanup EXIT INT TERM

# Start the proxy in the background and capture output
echo "Step 1: Starting litellm proxy with json_logs: true"
echo "Config file: reproduce_json_logs_config.yaml"
echo ""

# Start proxy and capture all output (stdout and stderr)
python3 start_proxy.py > proxy_output.log 2>&1 &
PROXY_PID=$!

echo "Proxy started with PID: $PROXY_PID"
echo "Waiting for proxy to start up..."
sleep 8

# Check if proxy is still running
if ! kill -0 $PROXY_PID 2>/dev/null; then
    echo "ERROR: Proxy failed to start!"
    echo "=== Proxy output ==="
    cat proxy_output.log
    exit 1
fi

echo "Proxy should be running. Checking logs..."
echo ""

# Show startup logs
echo "================================================================================"
echo "STARTUP LOGS (first 50 lines):"
echo "================================================================================"
head -50 proxy_output.log
echo ""
echo "... (truncated) ..."
echo ""

# Check if "Using json logs" message appears
if grep -q "Using json logs" proxy_output.log; then
    echo "✓ Found 'Using json logs' message in startup"
else
    echo "✗ Did NOT find 'Using json logs' message"
fi
echo ""

# Wait a bit more for full startup
sleep 2

echo "================================================================================"
echo "Step 2: Triggering errors by making requests"
echo "================================================================================"
echo ""

# Test 1: Make request to OpenAI model with fake key
echo "Test 1: Request to gpt-3.5-turbo (will fail with auth error)..."
curl -s -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234567890" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}]
  }' > /dev/null 2>&1 || true

sleep 2

# Test 2: Make request to Azure model
echo "Test 2: Request to azure-gpt-35 (will fail)..."
curl -s -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234567890" \
  -d '{
    "model": "azure-gpt-35",
    "messages": [{"role": "user", "content": "Hello"}]
  }' > /dev/null 2>&1 || true

sleep 2

# Test 3: Make request to Vertex AI model (will trigger credential error)
echo "Test 3: Request to vertex-gemini (will fail with credential error)..."
curl -s -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234567890" \
  -d '{
    "model": "vertex-gemini",
    "messages": [{"role": "user", "content": "Hello"}]
  }' > /dev/null 2>&1 || true

sleep 2

# Test 4: Check model info endpoint (triggers router model identification)
echo "Test 4: Request to /model/info (may trigger router errors)..."
curl -s -X GET "http://localhost:4000/model/info" \
  -H "Authorization: Bearer sk-1234567890" > /dev/null 2>&1 || true

sleep 2

# Test 5: Trigger health check if possible
echo "Test 5: Request to /health/liveliness..."
curl -s -X GET "http://localhost:4000/health/liveliness" > /dev/null 2>&1 || true

sleep 3

echo ""
echo "All test requests completed."
echo ""

# Analyze the logs
echo "================================================================================"
echo "Step 3: Analyzing logs for JSON vs plain text format"
echo "================================================================================"
echo ""

# Show all error logs
echo "=== ALL ERROR LOGS FROM PROXY ==="
echo ""
grep -i "error\|exception\|traceback\|failed" proxy_output.log | head -100 || echo "(No error logs found)"
echo ""

# Count JSON vs non-JSON log lines
JSON_COUNT=$(grep -E '^\{"message":.*"level":.*"timestamp":' proxy_output.log | wc -l)
PLAIN_TEXT_ERROR_COUNT=$(grep -E '^[0-9]{2}:[0-9]{2}:[0-9]{2}.*ERROR' proxy_output.log | wc -l)
TRACEBACK_COUNT=$(grep "^Traceback (most recent call last)" proxy_output.log | wc -l)

echo "================================================================================"
echo "LOG FORMAT ANALYSIS:"
echo "================================================================================"
echo "JSON formatted logs: $JSON_COUNT"
echo "Plain text ERROR logs: $PLAIN_TEXT_ERROR_COUNT"
echo "Plain text Tracebacks: $TRACEBACK_COUNT"
echo ""

if [ $PLAIN_TEXT_ERROR_COUNT -gt 0 ] || [ $TRACEBACK_COUNT -gt 0 ]; then
    echo "❌ BUG REPRODUCED! Found plain text logs when json_logs is enabled."
    echo ""
    echo "=== Examples of plain text ERROR logs ==="
    grep -E '^[0-9]{2}:[0-9]{2}:[0-9]{2}.*ERROR' proxy_output.log | head -5
    echo ""
    if [ $TRACEBACK_COUNT -gt 0 ]; then
        echo "=== Examples of plain text Tracebacks ==="
        grep -A 5 "^Traceback (most recent call last)" proxy_output.log | head -20
    fi
else
    echo "✓ All logs appear to be in JSON format."
fi

echo ""
echo "================================================================================"
echo "Full log file saved to: proxy_output.log"
echo "You can inspect it with: less proxy_output.log"
echo "================================================================================"
