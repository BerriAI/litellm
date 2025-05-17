---
id: spend_logs_deletion
title: Spend Logs Deletion
---

# Spend Log Cleanup

LiteLLM stores a log for every request. Over time, these logs can grow large and slow down your database. The Spend Log Cleanup feature helps manage database size by deleting old logs automatically.

---

## Usage

### Requirements

- **Postgres** (for log storage)
- **Redis** *(optional)* — required only if you're running multiple proxy instances and want to enable distributed locking

### Setup

Add this to your `proxy_config.yaml` under `general_settings`:

```yaml title="proxy_config.yaml"
general_settings:
  maximum_spend_logs_retention_period: "7d"  # Keep logs for 7 days

  # Optional: set how frequently cleanup should run
  maximum_spend_logs_retention_interval: "1d"  # Run cleanup every day

litellm_settings:
  cache: true
  cache_params:
    type: redis
```

### Configuration Options

#### `maximum_spend_logs_retention_period` (required)

How long logs should be kept before deletion. Supported formats:

- `"7d"` – 7 days
- `"24h"` – 24 hours
- `"60m"` – 60 minutes
- `"3600s"` – 3600 seconds

#### `maximum_spend_logs_retention_interval` (optional)

How often the cleanup job should run. Uses the same format as above. If not set, cleanup will run every 24 hours if and only if `maximum_spend_logs_retention_period` is set.

---

## How it works

### Step 1. Lock Acquisition (Optional with Redis)

If Redis is enabled, LiteLLM uses it to make sure only one instance runs the cleanup at a time.

- If the lock is acquired:
  - This instance proceeds with cleanup
  - Others skip it
- If no lock is present:
  - Cleanup still runs (useful for single-node setups)

![Working of spend log deletions](../../img/spend_log_deletion_working.png)  
*Working of spend log deletions*

---

### Step 2. Batch Deletion

Once cleanup starts:

- It calculates the cutoff date using the configured retention period
- Deletes logs older than the cutoff in **batches of 1000**
- Adds a short delay between batches to avoid overloading the database

### Default settings:
- **Batch size**: 1000 logs
- **Max batches per run**: 500
- **Max deletions per run**: 500,000 logs

You can change the number of batches using an environment variable:

```bash
SPEND_LOG_RUN_LOOPS=200
```

This would allow up to 200,000 logs to be deleted in one run.

![Batch deletion of old logs](../../img/spend_log_deletion_multi_pod.jpg)  
*Batch deletion of old logs*

---

## Summary

Spend Log Cleanup helps keep your database fast by regularly deleting old logs. It’s safe, customizable, and works well for both single-node and multi-node deployments.