-- AlterTable
ALTER TABLE "LiteLLM_EndUserTable" ADD COLUMN     "object_permission_id" TEXT;

-- AddForeignKey
ALTER TABLE "LiteLLM_EndUserTable" ADD CONSTRAINT "LiteLLM_EndUserTable_object_permission_id_fkey" FOREIGN KEY ("object_permission_id") REFERENCES "LiteLLM_ObjectPermissionTable"("object_permission_id") ON DELETE SET NULL ON UPDATE CASCADE;

