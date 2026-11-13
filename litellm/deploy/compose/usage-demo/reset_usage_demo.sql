-- Removes the usage-demo namespace rows from the disposable litellm-usage-demo database only.
DO $$
BEGIN
  IF current_setting('cluster_name') IS DISTINCT FROM 'litellm_usage_demo' THEN
    RAISE EXCEPTION 'refusing to run outside the litellm-usage-demo database (cluster_name is "%")',
      current_setting('cluster_name');
  END IF;
END
$$;

BEGIN;
DELETE FROM "LiteLLM_DailyUserSpend"
WHERE api_key LIKE 'usage-demo-key-%' AND user_id LIKE 'usage-demo-user-%';
DELETE FROM "LiteLLM_DailyTagSpend" WHERE api_key LIKE 'usage-demo-key-%';
DELETE FROM "LiteLLM_DailyGatewayRequests" WHERE route LIKE 'usage-demo:%';
DELETE FROM "LiteLLM_UserTable"
WHERE user_id IN ('usage-demo-user-a', 'usage-demo-user-b') AND user_alias LIKE 'Synthetic demo:%';
COMMIT;
