ALTER TABLE "LiteLLM_ProxyWorkerHeartbeat" ADD COLUMN IF NOT EXISTS "expires_at" TIMESTAMP(3);
