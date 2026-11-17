SET LOCAL lock_timeout = '10s';
ALTER TABLE "LiteLLM_SpendLogDelivery" ALTER COLUMN "delivery_id" DROP DEFAULT;
