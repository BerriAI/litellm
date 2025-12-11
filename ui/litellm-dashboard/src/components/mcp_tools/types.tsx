export interface Team {
  team_id: string;
  team_alias?: string;
  organization_id?: string | null;
}

// Default no auth value
export const AUTH_TYPE = {
  NONE: "none",
  API_KEY: "api_key",
  BEARER_TOKEN: "bearer_token",
  BASIC: "basic",
  OAUTH2: "oauth2",
};

export const TRANSPORT = {
  SSE: "sse",
  HTTP: "http",
};

export const handleTransport = (transport?: string | null): string => {
  console.log(transport);
  if (transport === null || transport === undefined) {
    return TRANSPORT.SSE;
  }

  return transport;
};

export const handleAuth = (authType?: string | null): string => {
  if (authType === null || authType === undefined) {
    return AUTH_TYPE.NONE;
  }

  return authType;
};

// Define the structure for tool input schema properties
export interface InputSchemaProperty {
  type: string;
  description?: string;
  properties?: Record<string, InputSchemaProperty>; // For nested object properties
  required?: string[]; // For required fields in nested objects
  enum?: string[]; // For enum values
  default?: any; // For default values
}

// Define the structure for the input schema of a tool
export interface InputSchema {
  type: "object";
  properties: Record<string, InputSchemaProperty>;
  required?: string[];
}

// Define MCPServerCostInfo for cost tracking
export interface MCPServerCostInfo {
  default_cost_per_query?: number | null;
  tool_name_to_cost_per_query?: Record<string, number | null>;
}

// Define MCP provider info
export interface MCPInfo {
  server_name: string;
  description?: string;
  logo_url?: string;
  mcp_server_cost_info?: MCPServerCostInfo | null;
}

// Define the structure for a single MCP tool
export interface MCPTool {
  name: string;
  description?: string;
  inputSchema: InputSchema | string; // API returns string "tool_input_schema" or the actual schema
  mcp_info: MCPInfo;
  // Function to select a tool (added in the component)
  onToolSelect?: (tool: MCPTool) => void;
}

// Define the response structure for the listMCPTools endpoint - now a flat array
export type ListMCPToolsResponse = MCPTool[];

// Define the argument structure for calling an MCP tool
export interface CallMCPToolArgs {
  name: string;
  arguments: Record<string, any> | null;
  server_name?: string; // Now using server_name from mcp_info
}

// Define the possible content types in the response
export interface MCPTextContent {
  type: "text";
  text: string;
  annotations?: any;
}

export interface MCPImageContent {
  type: "image";
  url?: string;
  data?: string;
}

export interface MCPEmbeddedResource {
  type: "embedded_resource";
  resource_type?: string;
  url?: string;
  data?: any;
}

// Define the union type for the content array in the response
export type MCPContent = MCPTextContent | MCPImageContent | MCPEmbeddedResource;

// Define the response structure for the callMCPTool endpoint
export type CallMCPToolResponse = {
  content: MCPContent[];
  _meta: any;
  isError: boolean;
  structuredContent: any;
};

// Props for the main component
export interface MCPToolsViewerProps {
  serverId: string;
  accessToken: string | null;
  auth_type?: string | null;
  userRole: string | null;
  userID: string | null;
  serverAlias?: string | null;
}

export interface MCPServer {
  server_id: string;
  server_name?: string | null;
  alias?: string | null;
  description?: string | null;
  url: string;
  transport?: string | null;
  auth_type?: string | null;
  mcp_info?: MCPInfo | null;
  created_at: string;
  created_by: string;
  updated_at: string;
  updated_by: string;
  extra_headers?: string[] | null;
  static_headers?: Record<string, string> | null;
  status?: "healthy" | "unhealthy" | "unknown";
  last_health_check?: string | null;
  health_check_error?: string | null;
  teams?: Team[];
  mcp_access_groups?: string[];
  allowed_tools?: string[];
}

export interface MCPServerProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}
