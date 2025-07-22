-- AlterTable
ALTER TABLE "LiteLLM_OrganizationTable" ADD COLUMN     "object_permission_id" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN     "object_permission_id" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_UserTable" ADD COLUMN     "object_permission_id" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN     "object_permission_id" TEXT;

-- CreateTable
CREATE TABLE "LiteLLM_ObjectPermissionTable" (
    "object_permission_id" TEXT NOT NULL,
    "mcp_servers" TEXT[] DEFAULT ARRAY[]::TEXT[],

    CONSTRAINT "LiteLLM_ObjectPermissionTable_pkey" PRIMARY KEY ("object_permission_id")
);

-- AddForeignKey
ALTER TABLE "LiteLLM_OrganizationTable" ADD CONSTRAINT "LiteLLM_OrganizationTable_object_permission_id_fkey" FOREIGN KEY ("object_permission_id") REFERENCES "LiteLLM_ObjectPermissionTable"("object_permission_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_TeamTable" ADD CONSTRAINT "LiteLLM_TeamTable_object_permission_id_fkey" FOREIGN KEY ("object_permission_id") REFERENCES "LiteLLM_ObjectPermissionTable"("object_permission_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_UserTable" ADD CONSTRAINT "LiteLLM_UserTable_object_permission_id_fkey" FOREIGN KEY ("object_permission_id") REFERENCES "LiteLLM_ObjectPermissionTable"("object_permission_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT "LiteLLM_VerificationToken_object_permission_id_fkey" FOREIGN KEY ("object_permission_id") REFERENCES "LiteLLM_ObjectPermissionTable"("object_permission_id") ON DELETE SET NULL ON UPDATE CASCADE;

