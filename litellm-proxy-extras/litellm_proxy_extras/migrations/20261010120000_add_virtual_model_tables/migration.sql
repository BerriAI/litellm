-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_VirtualModelTable" (
    "virtual_model_id" TEXT NOT NULL,
    "virtual_model_name" TEXT NOT NULL,
    "description" TEXT,
    "routing_strategy" TEXT NOT NULL DEFAULT 'ordered',
    "num_retries" INTEGER,
    "timeout" INTEGER,
    "stream_timeout" INTEGER,
    "allowed_fails" INTEGER,
    "cooldown_time" INTEGER,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_VirtualModelTable_pkey" PRIMARY KEY ("virtual_model_id")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_VirtualModelTable_virtual_model_name_key" ON "LiteLLM_VirtualModelTable"("virtual_model_name");

-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_VirtualModelMemberTable" (
    "virtual_model_id" TEXT NOT NULL,
    "model_id" TEXT NOT NULL,
    "priority" INTEGER NOT NULL DEFAULT 1,

    CONSTRAINT "LiteLLM_VirtualModelMemberTable_pkey" PRIMARY KEY ("virtual_model_id","model_id")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_VirtualModelMemberTable_model_id_idx" ON "LiteLLM_VirtualModelMemberTable"("model_id");

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_VirtualModelMemberTable_virtual_model_id_fkey') THEN
        ALTER TABLE "LiteLLM_VirtualModelMemberTable" ADD CONSTRAINT "LiteLLM_VirtualModelMemberTable_virtual_model_id_fkey" FOREIGN KEY ("virtual_model_id") REFERENCES "LiteLLM_VirtualModelTable"("virtual_model_id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_VirtualModelMemberTable_model_id_fkey') THEN
        ALTER TABLE "LiteLLM_VirtualModelMemberTable" ADD CONSTRAINT "LiteLLM_VirtualModelMemberTable_model_id_fkey" FOREIGN KEY ("model_id") REFERENCES "LiteLLM_ProxyModelTable"("model_id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;
