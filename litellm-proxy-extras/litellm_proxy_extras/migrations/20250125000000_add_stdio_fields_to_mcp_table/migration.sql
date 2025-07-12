-- Add stdio fields to existing MCP table
ALTER TABLE "LiteLLM_MCPServerTable" 
ADD COLUMN "command" TEXT,
ADD COLUMN "args" TEXT[],
ADD COLUMN "env" JSONB;

-- Make URL nullable for stdio transport
ALTER TABLE "LiteLLM_MCPServerTable" 
ALTER COLUMN "url" DROP NOT NULL;