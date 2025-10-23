/*
  Warnings:

  - You are about to drop the column `spec_version` on the `LiteLLM_MCPServerTable` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "LiteLLM_MCPServerTable" DROP COLUMN "spec_version";
