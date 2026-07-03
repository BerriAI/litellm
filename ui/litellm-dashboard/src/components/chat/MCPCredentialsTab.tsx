"use client";

/**
 * MCPCredentialsTab
 *
 * Shows all OAuth2 MCP connections the calling user has stored.
 * Lives in the Chat sidebar's "Credentials" tab.
 */

import React, { useState } from "react";
import { Spin, Table, Tag } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import MessageManager from "@/components/molecules/message_manager";
import { DeleteOutlined, LinkOutlined } from "@ant-design/icons";
import { deleteMCPOAuthUserCredential, listMCPUserCredentials, MCPUserCredentialListItem } from "../networking";

const MCP_CREDENTIALS_QUERY_KEY = "mcp-user-credentials";

interface Props {
  accessToken: string;
}

function relativeTime(isoString: string | null | undefined): string {
  if (!isoString) return "";
  try {
    const date = new Date(isoString);
    const diffMs = Date.now() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    if (diffSec < 60) return "just now";
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    return `${Math.floor(diffHr / 24)}d ago`;
  } catch {
    return "";
  }
}

function expiryLabel(isoString: string | null | undefined): string {
  if (!isoString) return "Does not expire";
  try {
    const exp = new Date(isoString);
    const diffMs = exp.getTime() - Date.now();
    if (diffMs <= 0) return "Expired";
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHr = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHr / 24);
    if (diffDay > 0) return `Expires in ${diffDay}d`;
    if (diffHr > 0) return `Expires in ${diffHr}h`;
    return `Expires in ${diffMin}m`;
  } catch {
    return "";
  }
}

const MCPCredentialsTab: React.FC<Props> = ({ accessToken }) => {
  const queryClient = useQueryClient();
  const [revoking, setRevoking] = useState<Set<string>>(new Set());

  const { data: credentials = [], isLoading: loading } = useQuery({
    queryKey: [MCP_CREDENTIALS_QUERY_KEY, accessToken],
    queryFn: () => listMCPUserCredentials(accessToken),
    enabled: !!accessToken,
  });

  const handleRevoke = async (serverId: string) => {
    setRevoking((prev) => new Set(prev).add(serverId));
    try {
      await deleteMCPOAuthUserCredential(accessToken, serverId);
      queryClient.setQueryData<MCPUserCredentialListItem[]>([MCP_CREDENTIALS_QUERY_KEY, accessToken], (prev) =>
        (prev ?? []).filter((c) => c.server_id !== serverId),
      );
    } catch {
      MessageManager.error("Failed to revoke connection. Please try again.");
    } finally {
      setRevoking((prev) => {
        const n = new Set(prev);
        n.delete(serverId);
        return n;
      });
    }
  };

  const displayName = (c: MCPUserCredentialListItem) => c.alias || c.server_name || c.server_id;

  const columns: ColumnsType<MCPUserCredentialListItem> = [
    {
      title: "App",
      key: "app",
      render: (_, cred) => <span className="text-sm font-medium text-gray-900">{displayName(cred)}</span>,
    },
    {
      title: "Connected",
      key: "connected",
      render: (_, cred) => <span className="text-sm text-gray-500">{relativeTime(cred.connected_at) || "—"}</span>,
    },
    {
      title: "Status",
      key: "status",
      render: (_, cred) => {
        const exp = expiryLabel(cred.expires_at);
        return <Tag color={exp === "Expired" ? "red" : "green"}>{exp}</Tag>;
      },
    },
    {
      title: "Actions",
      key: "actions",
      align: "right",
      render: (_, cred) => {
        const isRevoking = revoking.has(cred.server_id);
        return (
          <button
            onClick={() => handleRevoke(cred.server_id)}
            disabled={isRevoking}
            title="Revoke connection"
            className={`inline-flex items-center justify-center rounded-md border border-gray-200 px-2 py-1 text-gray-400 hover:text-red-500 hover:border-red-200 transition-colors ${isRevoking ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
            style={{ background: "none" }}
          >
            {isRevoking ? <Spin size="small" /> : <DeleteOutlined className="text-sm" />}
          </button>
        );
      },
    },
  ];

  return (
    <div className="w-full">
      {/* Header */}
      <div className="mb-4">
        <h2 className="text-base font-semibold text-gray-900 mb-0.5">App Credentials</h2>
        <p className="text-sm text-gray-500 m-0">Your stored OAuth connections — used automatically in chat.</p>
      </div>

      {loading ? (
        <div className="flex justify-center py-12">
          <Spin />
        </div>
      ) : credentials.length === 0 ? (
        <div className="text-center text-gray-400 text-sm py-12 border border-dashed border-gray-200 rounded-lg">
          <LinkOutlined className="text-2xl mb-3 block text-gray-300" />
          No connections yet.
          <br />
          Go to <strong>Apps</strong> and click <strong>Connect</strong> to authorize an MCP server.
        </div>
      ) : (
        <div className="rounded-lg border border-gray-200 overflow-hidden">
          <Table columns={columns} dataSource={credentials} rowKey="server_id" pagination={false} size="small" />
        </div>
      )}
    </div>
  );
};

export default MCPCredentialsTab;
