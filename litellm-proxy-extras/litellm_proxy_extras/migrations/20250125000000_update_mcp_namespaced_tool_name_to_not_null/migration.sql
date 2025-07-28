-- Update existing NULL values to empty string for mcp_namespaced_tool_name
UPDATE "LiteLLM_DailyUserSpend" 
SET mcp_namespaced_tool_name = '' 
WHERE mcp_namespaced_tool_name IS NULL;

UPDATE "LiteLLM_DailyTeamSpend" 
SET mcp_namespaced_tool_name = '' 
WHERE mcp_namespaced_tool_name IS NULL;

UPDATE "LiteLLM_DailyTagSpend" 
SET mcp_namespaced_tool_name = '' 
WHERE mcp_namespaced_tool_name IS NULL;

-- Update the SpendLogs table as well
UPDATE "LiteLLM_SpendLogs" 
SET mcp_namespaced_tool_name = '' 
WHERE mcp_namespaced_tool_name IS NULL;

-- Alter the columns to be NOT NULL with default empty string
ALTER TABLE "LiteLLM_DailyUserSpend" 
ALTER COLUMN mcp_namespaced_tool_name SET NOT NULL,
ALTER COLUMN mcp_namespaced_tool_name SET DEFAULT '';

ALTER TABLE "LiteLLM_DailyTeamSpend" 
ALTER COLUMN mcp_namespaced_tool_name SET NOT NULL,
ALTER COLUMN mcp_namespaced_tool_name SET DEFAULT '';

ALTER TABLE "LiteLLM_DailyTagSpend" 
ALTER COLUMN mcp_namespaced_tool_name SET NOT NULL,
ALTER COLUMN mcp_namespaced_tool_name SET DEFAULT ''; 