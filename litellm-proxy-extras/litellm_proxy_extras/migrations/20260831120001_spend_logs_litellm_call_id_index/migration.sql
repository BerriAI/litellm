-- The litellm_call_id index on LiteLLM_SpendLogs is built after migrate deploy, through
-- litellm_proxy_extras/request_log_indexes.py: concurrently on a plain table and per
-- partition on a partitioned one. The migration job builds it; a serving proxy that ran
-- the migrations itself builds it in the background once it serves. Postgres refuses
-- CREATE INDEX CONCURRENTLY on a partitioned parent, so this migration no longer runs it.
SELECT 1;
