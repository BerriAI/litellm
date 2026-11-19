-- DropForeignKey
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_LensReview_lens_id_fkey' AND conrelid = '"LiteLLM_LensReview"'::regclass) THEN
        ALTER TABLE "LiteLLM_LensReview" DROP CONSTRAINT "LiteLLM_LensReview_lens_id_fkey";
    END IF;
END $$;

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'LiteLLM_LensReview_lens_id_fkey' AND conrelid = '"LiteLLM_LensReview"'::regclass) THEN
        ALTER TABLE "LiteLLM_LensReview" ADD CONSTRAINT "LiteLLM_LensReview_lens_id_fkey" FOREIGN KEY ("lens_id") REFERENCES "LiteLLM_Lens"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;
