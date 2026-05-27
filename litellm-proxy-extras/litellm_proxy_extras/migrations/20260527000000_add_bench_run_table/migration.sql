-- CreateTable
CREATE TABLE "LiteLLM_BenchRun" (
    "bench_run_id" TEXT NOT NULL,
    "model_name" TEXT NOT NULL,
    "deployment_server" TEXT,
    "bench_type" TEXT,
    "input_tokens" INTEGER,
    "output_tokens" INTEGER,
    "max_concurrency" INTEGER,
    "raw_command" TEXT,
    "raw_results" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_BenchRun_pkey" PRIMARY KEY ("bench_run_id")
);

-- CreateIndex
CREATE INDEX "LiteLLM_BenchRun_model_name_idx" ON "LiteLLM_BenchRun"("model_name");
