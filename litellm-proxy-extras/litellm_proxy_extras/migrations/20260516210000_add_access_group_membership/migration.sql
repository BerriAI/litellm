-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_AccessGroupMembership" (
    "id" TEXT NOT NULL,
    "parent_group" TEXT NOT NULL,
    "child_group" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_AccessGroupMembership_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "LiteLLM_AccessGroupMembership_parent_group_child_group_key" ON "LiteLLM_AccessGroupMembership"("parent_group", "child_group");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_AccessGroupMembership_parent_group_idx" ON "LiteLLM_AccessGroupMembership"("parent_group");
