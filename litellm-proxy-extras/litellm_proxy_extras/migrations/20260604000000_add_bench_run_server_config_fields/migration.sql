-- AlterTable: add server-launch and speculative-decoding fields to LiteLLM_BenchRun
ALTER TABLE "LiteLLM_BenchRun" ADD COLUMN IF NOT EXISTS "max_num_batched_tokens" INTEGER;
ALTER TABLE "LiteLLM_BenchRun" ADD COLUMN IF NOT EXISTS "tensor_parallel_size" INTEGER;
ALTER TABLE "LiteLLM_BenchRun" ADD COLUMN IF NOT EXISTS "pipeline_parallel_size" INTEGER;
ALTER TABLE "LiteLLM_BenchRun" ADD COLUMN IF NOT EXISTS "data_parallel_size" INTEGER;
ALTER TABLE "LiteLLM_BenchRun" ADD COLUMN IF NOT EXISTS "kv_cache_dtype" TEXT;
ALTER TABLE "LiteLLM_BenchRun" ADD COLUMN IF NOT EXISTS "speculative_draft_model" TEXT;
ALTER TABLE "LiteLLM_BenchRun" ADD COLUMN IF NOT EXISTS "num_speculative_tokens" INTEGER;
