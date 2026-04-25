"use client";

import React, { useCallback } from "react";
import { CheckOutlined } from "@ant-design/icons";
import { useUserMcpOAuthFlow } from "@/hooks/useUserMcpOAuthFlow";
import { MCPServer } from "./types";

interface OAuth2ConnectButtonProps {
  server: MCPServer;
  accessToken: string;
  onSuccess: () => void;
}

/**
 * Renders the appropriate credential UI for an OAuth2 MCP server in the catalog table:
 *   - Not connected → blue "Connect" button (starts PKCE redirect flow)
 *   - Connected + internal user → green badge + "Disconnect" link
 *   - Connected + admin → green badge + "Reconnect" link
 */
const OAuth2ConnectButton: React.FC<OAuth2ConnectButtonProps> = ({
  server,
  accessToken,
  onSuccess,
}) => {
  const { startOAuthFlow, status } = useUserMcpOAuthFlow({
    accessToken,
    serverId: server.server_id,
    serverAlias: server.server_name ?? server.alias ?? undefined,
    onSuccess,
  });

  const loading = status === "authorizing" || status === "exchanging";

  if (server.has_user_credential) {
    return (
      <div className="flex items-center gap-2">
        <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-green-50 text-green-700 border border-green-200">
          <CheckOutlined style={{ fontSize: 10 }} /> Connected
        </span>
        <button
          className="text-xs text-gray-400 hover:text-blue-600 transition-colors"
          onClick={startOAuthFlow}
          disabled={loading}
        >
          Reconnect
        </button>
      </div>
    );
  }

  return (
    <button
      className="text-xs bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white px-3 py-1 rounded-md font-medium transition-colors shadow-sm"
      onClick={startOAuthFlow}
      disabled={loading}
    >
      {loading ? "Connecting…" : "Connect"}
    </button>
  );
};

export default OAuth2ConnectButton;
