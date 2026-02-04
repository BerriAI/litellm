# Troubleshooting: DB Connection Failures ("All connection attempts failed")

This guide addresses the database connection issues reported in [Issue #19921](https://github.com/BerriAI/litellm/issues/19921), where users experience:

```
LiteLLM Prisma Client Exception get_generic_data: All connection attempts failed
```

```json
{
    "error": {
        "message": "Authentication Error, All connection attempts failed",
        "type": "auth_error",
        "param": "None",
        "code": "401"
    }
}
```

## Root Causes

### 1. Per-Request Router Creation with Expensive Prometheus Initialization (v1.81.0-v1.81.5)

**Issue:** When `router_settings` exist in the database (global, team, or key level), the system was creating a new `Router` instance for every request. The `Router.__init__` was initializing `PrometheusServicesLogger` which called `REGISTRY.collect()` (O(n) operation) to check if metrics were registered. With many metrics, this became extremely CPU-intensive.

**Fixed in:** PR #20087 - Changed `is_metric_registered()` to use `REGISTRY._names_to_collectors` (O(1)) instead of `REGISTRY.collect()` (O(n))

### 2. Redundant Router Creation for Global Settings

**Issue:** Even after the Prometheus fix, global `router_settings` from DB were triggering per-request Router creation, but global settings are already applied to the shared `llm_router` at startup.

**Fixed in:** PR #20133 - Skip global settings in hierarchical lookup, only use key/team-specific settings for per-request Router creation.

### 3. DB Connection Pool Exhaustion (IAM Token Auth)

**Issue:** When using `iam_token_db_auth`, the IAM token refresh was creating new Prisma clients without properly disconnecting old ones. This led to connection pool exhaustion over time (6-12 hours).

**Fixed in:** PR #13140 - Ensure stale Prisma clients properly disconnect before reconnecting.

### 4. Connection Pool Limits Under High Load

The default `database_connection_pool_limit` is 10 per worker. Under high load with many concurrent requests, especially combined with the per-request Router creation issue, connections can be exhausted quickly.

## Solutions

### 1. Upgrade to Latest Version

Upgrade to **v1.80.16** or **v1.81.7+** which include all the performance fixes.

```bash
pip install litellm==1.81.7
```

### 2. Clean Up Stale Router Settings from Database

Check for and remove stale `router_settings` from your database:

```sql
-- Check for global router_settings
SELECT param_name, param_value FROM "LiteLLM_Config" WHERE param_name = 'router_settings';

-- Check for team router_settings
SELECT team_id, router_settings FROM "LiteLLM_TeamTable"
WHERE router_settings IS NOT NULL AND router_settings != '{}';

-- Check for key router_settings
SELECT token, router_settings FROM "LiteLLM_VerificationToken"
WHERE router_settings IS NOT NULL AND router_settings != '{}';

-- Remove global router_settings if not needed
DELETE FROM "LiteLLM_Config" WHERE param_name = 'router_settings';
```

### 3. Increase Connection Pool Settings

If you're still experiencing connection issues under high load, increase the connection pool settings:

```yaml
general_settings:
  database_connection_pool_limit: 50  # Increase from default 10
  database_connection_pool_timeout: 120  # Increase from default 60
```

Or via environment variables:
```bash
DATABASE_CONNECTION_POOL_LIMIT=50
DATABASE_CONNECTION_POOL_TIMEOUT=120
```

### 4. Enable Resilience for DB Unavailability

To allow requests to proceed even when the database is temporarily unavailable:

```yaml
general_settings:
  allow_requests_on_db_unavailable: true
```

### 5. Monitor Connection Usage

Check your PostgreSQL connection usage:

```sql
-- Check current connections
SELECT count(*) FROM pg_stat_activity WHERE datname = 'your_database';

-- Check max connections setting
SHOW max_connections;

-- View connection details
SELECT pid, usename, application_name, client_addr, state, query_start
FROM pg_stat_activity
WHERE datname = 'your_database'
ORDER BY query_start DESC;
```

## Diagnostic Steps

1. **Check LiteLLM version:**
   ```bash
   litellm --version
   ```

2. **Check `/health/readiness` endpoint:**
   ```bash
   curl http://localhost:4000/health/readiness
   ```

3. **Enable debug logging:**
   ```yaml
   general_settings:
     debug: true
   ```

4. **Check for router_settings in requests:**
   Look for `user_config` being passed in your request metadata, which triggers per-request Router creation.

## Environment-Specific Considerations

### AWS RDS with IAM Authentication

If using `IAM_TOKEN_DB_AUTH=true`:
- Ensure you're on v1.81.5+ which includes the token refresh fixes
- The token refresh happens automatically 3 minutes before expiration
- If you see connection exhaustion over time, ensure you've upgraded

### High-Availability Deployments

For deployments with multiple pods:
- Each pod has its own connection pool
- Total connections = `database_connection_pool_limit × number_of_workers × number_of_pods`
- Ensure your PostgreSQL `max_connections` can accommodate this

## Related Issues and PRs

- [Issue #19921](https://github.com/BerriAI/litellm/issues/19921) - Performance regression report
- [PR #20087](https://github.com/BerriAI/litellm/pull/20087) - Fix Prometheus O(n) scan
- [PR #20133](https://github.com/BerriAI/litellm/pull/20133) - Skip global router_settings in per-request Router
- [PR #13140](https://github.com/BerriAI/litellm/pull/13140) - Fix stale Prisma client disconnection
