-- Removes only what seed_usage_demo.sql wrote.
BEGIN;
DELETE FROM "LiteLLM_DailyUserSpend"
WHERE api_key LIKE 'usage-demo-key-%' AND user_id LIKE 'usage-demo-user-%';
DELETE FROM "LiteLLM_DailyGatewayRequests" WHERE route LIKE 'usage-demo:%';
DELETE FROM "LiteLLM_UserTable"
WHERE user_id IN ('usage-demo-user-a', 'usage-demo-user-b');
COMMIT;
