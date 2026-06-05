-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_CasbinRule" (
    "id" SERIAL NOT NULL,
    "ptype" TEXT NOT NULL,
    "v0" TEXT,
    "v1" TEXT,
    "v2" TEXT,
    "v3" TEXT,
    "v4" TEXT,
    "v5" TEXT,

    CONSTRAINT "LiteLLM_CasbinRule_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_CasbinRule_ptype_idx" ON "LiteLLM_CasbinRule"("ptype");
