CREATE TYPE "PublicRelayAccountStatus" AS ENUM ('INVITED', 'ACTIVE', 'FROZEN', 'CLOSED');
CREATE TYPE "PublicRelayLedgerEntryType" AS ENUM ('RESERVE', 'RELEASE', 'USAGE', 'ADJUSTMENT');
CREATE TYPE "PublicRelayReservationStatus" AS ENUM ('OPEN', 'FINALIZED', 'RELEASED');
CREATE TYPE "PublicRelayAuthTokenPurpose" AS ENUM ('ACTIVATION', 'PASSWORD_RESET');

CREATE TABLE "LiteLLM_PublicRelayAccount" (
    "account_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "normalized_email" TEXT NOT NULL,
    "company_name" TEXT NOT NULL,
    "notes" TEXT,
    "status" "PublicRelayAccountStatus" NOT NULL DEFAULT 'INVITED',
    "activated_at" TIMESTAMP(3),
    "session_version" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_PublicRelayAccount_pkey" PRIMARY KEY ("account_id")
);

CREATE TABLE "LiteLLM_PublicRelayWallet" (
    "wallet_id" TEXT NOT NULL,
    "account_id" TEXT NOT NULL,
    "currency" TEXT NOT NULL DEFAULT 'USD',
    "available_micros" BIGINT NOT NULL DEFAULT 0,
    "reserved_micros" BIGINT NOT NULL DEFAULT 0,
    "version" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_PublicRelayWallet_pkey" PRIMARY KEY ("wallet_id"),
    CONSTRAINT "LiteLLM_PublicRelayWallet_nonnegative" CHECK (
        "available_micros" >= 0 AND "reserved_micros" >= 0
    )
);

CREATE TABLE "LiteLLM_PublicRelayLedgerEntry" (
    "entry_id" TEXT NOT NULL,
    "wallet_id" TEXT NOT NULL,
    "entry_type" "PublicRelayLedgerEntryType" NOT NULL,
    "amount_micros" BIGINT NOT NULL,
    "available_after_micros" BIGINT NOT NULL,
    "reserved_after_micros" BIGINT NOT NULL,
    "idempotency_key" TEXT NOT NULL,
    "request_id" TEXT,
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_PublicRelayLedgerEntry_pkey" PRIMARY KEY ("entry_id")
);

CREATE TABLE "LiteLLM_PublicRelayModelPrice" (
    "price_id" TEXT NOT NULL,
    "model_name" TEXT NOT NULL,
    "version" INTEGER NOT NULL,
    "input_micros_per_million" BIGINT NOT NULL,
    "cached_input_micros_per_million" BIGINT,
    "output_micros_per_million" BIGINT,
    "embedding_micros_per_million" BIGINT,
    "default_max_output_tokens" INTEGER NOT NULL DEFAULT 4096,
    "max_output_tokens" INTEGER NOT NULL DEFAULT 4096,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "effective_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    CONSTRAINT "LiteLLM_PublicRelayModelPrice_pkey" PRIMARY KEY ("price_id"),
    CONSTRAINT "LiteLLM_PublicRelayModelPrice_rates_nonnegative" CHECK (
        "input_micros_per_million" >= 0
        AND ("cached_input_micros_per_million" IS NULL OR "cached_input_micros_per_million" >= 0)
        AND ("output_micros_per_million" IS NULL OR "output_micros_per_million" >= 0)
        AND ("embedding_micros_per_million" IS NULL OR "embedding_micros_per_million" >= 0)
        AND "default_max_output_tokens" > 0
        AND "max_output_tokens" >= "default_max_output_tokens"
    )
);

CREATE TABLE "LiteLLM_PublicRelayReservation" (
    "reservation_id" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "account_id" TEXT NOT NULL,
    "wallet_id" TEXT NOT NULL,
    "price_id" TEXT NOT NULL,
    "reserved_micros" BIGINT NOT NULL,
    "input_tokens" INTEGER NOT NULL,
    "max_output_tokens" INTEGER NOT NULL,
    "status" "PublicRelayReservationStatus" NOT NULL DEFAULT 'OPEN',
    "expires_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_PublicRelayReservation_pkey" PRIMARY KEY ("reservation_id")
);

CREATE TABLE "LiteLLM_PublicRelayRequestCharge" (
    "charge_id" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "account_id" TEXT NOT NULL,
    "price_id" TEXT NOT NULL,
    "input_tokens" INTEGER NOT NULL,
    "cached_input_tokens" INTEGER NOT NULL DEFAULT 0,
    "output_tokens" INTEGER NOT NULL DEFAULT 0,
    "charged_micros" BIGINT NOT NULL,
    "upstream_cost_micros" BIGINT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_PublicRelayRequestCharge_pkey" PRIMARY KEY ("charge_id")
);

CREATE TABLE "LiteLLM_PublicRelaySession" (
    "session_id" TEXT NOT NULL,
    "token_hash" TEXT NOT NULL,
    "account_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "normalized_email" TEXT NOT NULL,
    "session_version" INTEGER NOT NULL,
    "csrf_token" TEXT NOT NULL,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_PublicRelaySession_pkey" PRIMARY KEY ("session_id")
);

CREATE TABLE "LiteLLM_PublicRelayAuthToken" (
    "auth_token_id" TEXT NOT NULL,
    "token_hash" TEXT NOT NULL,
    "account_id" TEXT NOT NULL,
    "purpose" "PublicRelayAuthTokenPurpose" NOT NULL,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "consumed_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_PublicRelayAuthToken_pkey" PRIMARY KEY ("auth_token_id")
);

CREATE TABLE "LiteLLM_PublicRelayRateLimit" (
    "rate_limit_id" TEXT NOT NULL,
    "key_hash" TEXT NOT NULL,
    "window_started_at" TIMESTAMP(3) NOT NULL,
    "window_seconds" INTEGER NOT NULL,
    "count" INTEGER NOT NULL DEFAULT 1,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_PublicRelayRateLimit_pkey" PRIMARY KEY ("rate_limit_id")
);

CREATE TABLE "LiteLLM_PublicRelayRequestContent" (
    "content_id" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "account_id" TEXT NOT NULL,
    "key_version" INTEGER NOT NULL,
    "nonce_b64" TEXT NOT NULL,
    "ciphertext_b64" TEXT NOT NULL,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_PublicRelayRequestContent_pkey" PRIMARY KEY ("content_id")
);

CREATE UNIQUE INDEX "LiteLLM_PublicRelayAccount_user_id_key" ON "LiteLLM_PublicRelayAccount"("user_id");
CREATE UNIQUE INDEX "LiteLLM_PublicRelayAccount_normalized_email_key" ON "LiteLLM_PublicRelayAccount"("normalized_email");
CREATE INDEX "LiteLLM_PublicRelayAccount_status_idx" ON "LiteLLM_PublicRelayAccount"("status");
CREATE INDEX "LiteLLM_PublicRelayAccount_created_at_idx" ON "LiteLLM_PublicRelayAccount"("created_at");
CREATE UNIQUE INDEX "LiteLLM_PublicRelayWallet_account_id_key" ON "LiteLLM_PublicRelayWallet"("account_id");
CREATE UNIQUE INDEX "LiteLLM_PublicRelayLedgerEntry_idempotency_key_key" ON "LiteLLM_PublicRelayLedgerEntry"("idempotency_key");
CREATE INDEX "LiteLLM_PublicRelayLedgerEntry_wallet_id_created_at_idx" ON "LiteLLM_PublicRelayLedgerEntry"("wallet_id", "created_at");
CREATE INDEX "LiteLLM_PublicRelayLedgerEntry_request_id_idx" ON "LiteLLM_PublicRelayLedgerEntry"("request_id");
CREATE UNIQUE INDEX "LiteLLM_PublicRelayModelPrice_model_name_version_key" ON "LiteLLM_PublicRelayModelPrice"("model_name", "version");
CREATE INDEX "LiteLLM_PublicRelayModelPrice_model_name_enabled_effective_idx" ON "LiteLLM_PublicRelayModelPrice"("model_name", "enabled", "effective_at");
CREATE UNIQUE INDEX "LiteLLM_PublicRelayReservation_request_id_key" ON "LiteLLM_PublicRelayReservation"("request_id");
CREATE INDEX "LiteLLM_PublicRelayReservation_status_expires_at_idx" ON "LiteLLM_PublicRelayReservation"("status", "expires_at");
CREATE INDEX "LiteLLM_PublicRelayReservation_account_id_created_at_idx" ON "LiteLLM_PublicRelayReservation"("account_id", "created_at");
CREATE UNIQUE INDEX "LiteLLM_PublicRelayRequestCharge_request_id_key" ON "LiteLLM_PublicRelayRequestCharge"("request_id");
CREATE INDEX "LiteLLM_PublicRelayRequestCharge_account_id_created_at_idx" ON "LiteLLM_PublicRelayRequestCharge"("account_id", "created_at");
CREATE UNIQUE INDEX "LiteLLM_PublicRelaySession_token_hash_key" ON "LiteLLM_PublicRelaySession"("token_hash");
CREATE INDEX "LiteLLM_PublicRelaySession_account_id_expires_at_idx" ON "LiteLLM_PublicRelaySession"("account_id", "expires_at");
CREATE INDEX "LiteLLM_PublicRelaySession_expires_at_idx" ON "LiteLLM_PublicRelaySession"("expires_at");
CREATE UNIQUE INDEX "LiteLLM_PublicRelayAuthToken_token_hash_key" ON "LiteLLM_PublicRelayAuthToken"("token_hash");
CREATE INDEX "LiteLLM_PublicRelayAuthToken_account_id_purpose_consumed_idx" ON "LiteLLM_PublicRelayAuthToken"("account_id", "purpose", "consumed_at");
CREATE INDEX "LiteLLM_PublicRelayAuthToken_expires_at_idx" ON "LiteLLM_PublicRelayAuthToken"("expires_at");
CREATE UNIQUE INDEX "LiteLLM_PublicRelayRateLimit_window_key" ON "LiteLLM_PublicRelayRateLimit"("key_hash", "window_started_at", "window_seconds");
CREATE INDEX "LiteLLM_PublicRelayRateLimit_expires_at_idx" ON "LiteLLM_PublicRelayRateLimit"("expires_at");
CREATE UNIQUE INDEX "LiteLLM_PublicRelayRequestContent_request_id_key" ON "LiteLLM_PublicRelayRequestContent"("request_id");
CREATE INDEX "LiteLLM_PublicRelayRequestContent_expires_at_idx" ON "LiteLLM_PublicRelayRequestContent"("expires_at");
CREATE INDEX "LiteLLM_PublicRelayRequestContent_account_id_created_at_idx" ON "LiteLLM_PublicRelayRequestContent"("account_id", "created_at");

ALTER TABLE "LiteLLM_PublicRelayAccount" ADD CONSTRAINT "LiteLLM_PublicRelayAccount_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "LiteLLM_UserTable"("user_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "LiteLLM_PublicRelayWallet" ADD CONSTRAINT "LiteLLM_PublicRelayWallet_account_id_fkey" FOREIGN KEY ("account_id") REFERENCES "LiteLLM_PublicRelayAccount"("account_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "LiteLLM_PublicRelayLedgerEntry" ADD CONSTRAINT "LiteLLM_PublicRelayLedgerEntry_wallet_id_fkey" FOREIGN KEY ("wallet_id") REFERENCES "LiteLLM_PublicRelayWallet"("wallet_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "LiteLLM_PublicRelayReservation" ADD CONSTRAINT "LiteLLM_PublicRelayReservation_account_id_fkey" FOREIGN KEY ("account_id") REFERENCES "LiteLLM_PublicRelayAccount"("account_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "LiteLLM_PublicRelayReservation" ADD CONSTRAINT "LiteLLM_PublicRelayReservation_wallet_id_fkey" FOREIGN KEY ("wallet_id") REFERENCES "LiteLLM_PublicRelayWallet"("wallet_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "LiteLLM_PublicRelayReservation" ADD CONSTRAINT "LiteLLM_PublicRelayReservation_price_id_fkey" FOREIGN KEY ("price_id") REFERENCES "LiteLLM_PublicRelayModelPrice"("price_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "LiteLLM_PublicRelayRequestCharge" ADD CONSTRAINT "LiteLLM_PublicRelayRequestCharge_account_id_fkey" FOREIGN KEY ("account_id") REFERENCES "LiteLLM_PublicRelayAccount"("account_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "LiteLLM_PublicRelayRequestCharge" ADD CONSTRAINT "LiteLLM_PublicRelayRequestCharge_price_id_fkey" FOREIGN KEY ("price_id") REFERENCES "LiteLLM_PublicRelayModelPrice"("price_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "LiteLLM_PublicRelaySession" ADD CONSTRAINT "LiteLLM_PublicRelaySession_account_id_fkey" FOREIGN KEY ("account_id") REFERENCES "LiteLLM_PublicRelayAccount"("account_id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "LiteLLM_PublicRelayAuthToken" ADD CONSTRAINT "LiteLLM_PublicRelayAuthToken_account_id_fkey" FOREIGN KEY ("account_id") REFERENCES "LiteLLM_PublicRelayAccount"("account_id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "LiteLLM_PublicRelayRequestContent" ADD CONSTRAINT "LiteLLM_PublicRelayRequestContent_account_id_fkey" FOREIGN KEY ("account_id") REFERENCES "LiteLLM_PublicRelayAccount"("account_id") ON DELETE RESTRICT ON UPDATE CASCADE;
