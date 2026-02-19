# Mavvrik Integration Setup

Enable Mavvrik cost tracking in your LiteLLM proxy with three steps: add configuration, restart the proxy, verify the integration works.

## Prerequisites

You need:
- LiteLLM proxy running with PostgreSQL database
- Mavvrik credentials from your account representative:
  - Tenant ID
  - Instance ID
  - API endpoint URL
  - API key

## Step 1: Add Mavvrik Configuration

Add your Mavvrik credentials to the proxy configuration file.

**In `proxy_server_config.yaml` or your config file:**

```yaml
litellm_settings:
  success_callback: ["mavvrik"]  # Logs successful API calls only (recommended)
  # Alternative: callbacks: ["mavvrik"]  # Logs both success and failure

environment_variables:
  MAVVRIK_TENANT_ID: "your-tenant-id"
  MAVVRIK_INSTANCE_ID: "your-instance-id"
  MAVVRIK_ENDPOINT: "https://api.mavvrik.io"  # Or your endpoint
  MAVVRIK_API_KEY: "your-api-key"
```

**Security note:** Store credentials in environment variables or a secrets manager, not directly in the config file.

## Step 2: Restart the Proxy

Restart your LiteLLM proxy to load the new configuration.

```bash
# Stop the running proxy, then start it with your config
litellm --config /path/to/proxy_server_config.yaml
```

**On startup**, the proxy logs confirm Mavvrik loaded:

```
Initialized Success Callbacks - ['mavvrik']
# Or if using callbacks: ['mavvrik']
```

**Note:** Use `success_callback` to log only successful API calls (recommended for cost tracking). Use `callbacks` to log both successful and failed calls.

## Step 3: Verify the Integration

Check that the Mavvrik scheduler started.

**Run this command:**

```bash
curl http://localhost:4000/health/readiness
```

**Look for** `mavvrik_scheduler_running: true` in the response.

## What Happens Next

The integration runs automatically:

1. **Every hour**, the scheduler exports yesterday's usage data to Mavvrik
2. **On first run**, the scheduler backfills all historical data from your database
3. **Each export** uploads one file per calendar day to Google Cloud Storage
4. **If Mavvrik resets the cursor**, the next export honors the reset

You do nothing. The integration handles everything.

## Troubleshooting

**If the scheduler does not start:**
- Check the proxy logs for errors about Mavvrik credentials
- Verify all four environment variables are set correctly
- Confirm your database connection works (Mavvrik requires PostgreSQL)

**If data does not appear in Mavvrik:**
- Wait up to 60 minutes for the first scheduled export
- Check the proxy logs for export errors
- Verify your Mavvrik API key has write permissions

**For immediate testing:**
- Use the admin endpoint to trigger a manual export:
  ```bash
  curl -X POST http://localhost:4000/mavvrik/export \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d '{"date_str": "2026-02-18"}'
  ```

## Support

Contact your Mavvrik account representative or LiteLLM support if you encounter issues.
