-- AlterTable
ALTER TABLE "LiteLLM_ManagedFileTable" ADD COLUMN IF NOT EXISTS "created_by" TEXT,
ADD COLUMN IF NOT EXISTS "flat_model_file_ids" TEXT[] DEFAULT ARRAY[]::TEXT[],
ADD COLUMN IF NOT EXISTS "updated_by" TEXT;

-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_ManagedObjectTable" (
    "id" TEXT NOT NULL,
    "unified_object_id" TEXT NOT NULL,
    "model_object_id" TEXT NOT NULL,
    "file_object" JSONB NOT NULL,
    "file_purpose" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_ManagedObjectTable_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_ManagedObjectTable_unified_object_id_key" ON "LiteLLM_ManagedObjectTable"("unified_object_id");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_ManagedObjectTable_model_object_id_key" ON "LiteLLM_ManagedObjectTable"("model_object_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedObjectTable_unified_object_id_idx" ON "LiteLLM_ManagedObjectTable"("unified_object_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_ManagedObjectTable_model_object_id_idx" ON "LiteLLM_ManagedObjectTable"("model_object_id");

