# Data Leak Test Instructions

## Overview
This test reproduces the cross-session data contamination issue where users receive data from other users' sessions.

## Prerequisites
1. LiteLLM proxy running on `localhost:4000`
2. API key `sk-1234` configured
3. Model `openai/my-fake-model` available
4. Python packages: `locust`, `requests`

## Step 1: Verify Setup
```bash
python test_setup.py
```

This will test:
- ✅ Proxy health endpoint
- ✅ Memory monitoring endpoint  
- ✅ Chat completions with your model

## Step 2: Run Data Leak Test

### Option A: Command Line (Headless)
```bash
locust -f test_data_leak_simple.py --host=http://localhost:4000 --headless -u 20 -r 5 -t 3m
```

Parameters:
- `-u 20`: 20 concurrent users
- `-r 5`: Spawn 5 users per second
- `-t 3m`: Run for 3 minutes

### Option B: Web UI (Recommended)
```bash
locust -f test_data_leak_simple.py --host=http://localhost:4000
```

Then open http://localhost:8089 and configure:
- **Number of users**: 20
- **Spawn rate**: 5 users/second
- **Host**: http://localhost:4000

## Step 3: Monitor Results

### Real-time Monitoring
- Watch console output for: `🚨 DATA LEAK! User X got data from Y`
- Monitor memory usage in web UI or logs

### Files Generated
- `leak_detected.json` - Created immediately when leak detected
- `final_leak_test_report.json` - Complete test summary

## What the Test Does

1. **Creates unique users**: Each simulated user gets a unique ID and secret data
2. **Sends chat requests**: Users send messages containing their secret data
3. **Checks for contamination**: Validates that responses don't contain other users' secrets
4. **Monitors memory**: Tracks memory usage throughout the test
5. **Reports results**: Generates detailed leak detection reports

## Expected Results

### ✅ BEFORE Fix (Broken Version)
```
🚨 CRITICAL: DATA LEAKS DETECTED!
  User abc12345 received secret from def67890
  Leaked: SECRET_def67890_1234567890
```

### ✅ AFTER Fix (Fixed Version)  
```
✅ NO DATA LEAKS DETECTED
User sessions appear properly isolated
```

## Troubleshooting

### Connection Issues
- Verify proxy is running: `curl http://localhost:4000/health`
- Check API key: `curl -H "Authorization: Bearer sk-1234" http://localhost:4000/health`

### Model Issues
- Verify model exists in your proxy config
- Check proxy logs for model loading errors

### Memory Endpoint Issues
- Memory endpoint might require admin permissions
- Check if `/health/memory` vs `/v1/health/memory` is correct for your setup

## Analysis

### Memory Leaks
- Compare start vs end memory usage
- Look for continuously growing memory

### Data Leaks  
- Any detection of cross-user data = CRITICAL security issue
- Check `leak_detected.json` for immediate leak alerts

Run this test before and after applying the fix to verify the security issue is resolved.