-- Add health check fields to MCP server table
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN "status" TEXT DEFAULT 'unknown';
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN "last_health_check" TIMESTAMP(3);
ALTER TABLE "LiteLLM_MCPServerTable" ADD COLUMN "health_check_error" TEXT; 