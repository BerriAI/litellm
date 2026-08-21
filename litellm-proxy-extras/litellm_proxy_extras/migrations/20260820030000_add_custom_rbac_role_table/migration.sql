-- CreateTable
CREATE TABLE "LiteLLM_CustomRBACRoleTable" (
    "role_name" TEXT NOT NULL,
    "description" TEXT,
    "allowed_routes" TEXT[],
    "inherits" TEXT[],
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_CustomRBACRoleTable_pkey" PRIMARY KEY ("role_name")
);
