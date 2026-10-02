import { useUISettings } from "./useUISettings";

export const STDIO_MCP_SETTING_KEY = "enable_stdio_mcp";

export const useStdioMcpEnabled = (): boolean => useUISettings().data?.values?.[STDIO_MCP_SETTING_KEY] === true;
