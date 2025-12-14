import { mcpToolsCall } from "../../networking";

export interface MCPTool {
  name: string;
  description: string;
  title: string | null;
  inputSchema: {
    type: string;
    properties: Record<string, any>;
    required: string[];
  };
  outputSchema: any;
  annotations: any;
  _meta: any;
}

interface MCPToolsResponse {
  tools: MCPTool[];
}

export async function fetchAvailableMCPTools(accessToken: string): Promise<MCPTool[]> {
  try {
    const data = (await mcpToolsCall(accessToken)) as MCPToolsResponse;
    return data.tools || [];
  } catch (error) {
    console.error("Error fetching MCP tools:", error);
    return [];
  }
}
