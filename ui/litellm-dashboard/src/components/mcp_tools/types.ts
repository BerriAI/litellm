export interface Team {
  team_id: string;
  team_alias?: string;
  organization_id?: string | null;
}

export interface MCPServer {
  server_id: string;
  alias: string;
  description?: string | null;
  url: string;
  transport: string;
  spec_version: string;
  auth_type: string;
  created_at: string;
  created_by: string;
  updated_at: string;
  updated_by: string;
  teams?: Team[];
}

export interface MCPServerProps {
  accessToken: string;
  userRole: string;
  userID: string;
}

export const handleAuth = (auth_type: string) => {
  switch (auth_type) {
    case "bearer":
      return "Bearer Token";
    case "basic":
      return "Basic Auth";
    default:
      return auth_type;
  }
};

export const handleTransport = (transport: string) => {
  switch (transport) {
    case "http":
      return "HTTP";
    case "https":
      return "HTTPS";
    default:
      return transport;
  }
}; 