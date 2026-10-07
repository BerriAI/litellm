-- DropForeignKey
ALTER TABLE "LiteLLM_LensReview" DROP CONSTRAINT IF EXISTS "LiteLLM_LensReview_lens_id_fkey";

-- AddForeignKey
ALTER TABLE "LiteLLM_LensReview" ADD CONSTRAINT "LiteLLM_LensReview_lens_id_fkey" FOREIGN KEY ("lens_id") REFERENCES "LiteLLM_Lens"("id") ON DELETE CASCADE ON UPDATE CASCADE;
