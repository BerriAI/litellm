"use client";

/**
 * MCPCredentialsTab
 *
 * Shows all OAuth2 MCP connections the calling user has stored.
 * Lives in the Chat sidebar's "Credentials" tab.
 */

import React, { useCallback, useEffect, useState } from "react";
import { Spin, message } from "antd";
import { CheckCircleOutlined, DeleteOutlined, LinkOutlined } from "@ant-design/icons";
import {
  deleteMCPOAuthUserCredential,
  listMCPUserCredentials,
  MCPUserCredentialListItem,
} from "../networking";

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
  const [credentials, setCredentials] = useState<MCPUserCredentialListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [revoking, setRevoking] = useState<Set<string>>(new Set());

  const load = useCallback(() => {
    setLoading(true);
    listMCPUserCredentials(accessToken)
      .then(setCredentials)
      .catch(() => setCredentials([]))
      .finally(() => setLoading(false));
  }, [accessToken]);

  useEffect(() => { load(); }, [load]);

  const handleRevoke = async (serverId: string) => {
    setRevoking((prev) => new Set(prev).add(serverId));
    try {
      await deleteMCPOAuthUserCredential(accessToken, serverId);
      setCredentials((prev) => prev.filter((c) => c.server_id !== serverId));
    } catch {
      message.error("Failed to revoke connection. Please try again.");
    } finally {
      setRevoking((prev) => { const n = new Set(prev); n.delete(serverId); return n; });
    }
  };

  const displayName = (c: MCPUserCredentialListItem) =>
    c.alias || c.server_name || c.server_id;

  return (
    <div style={{ width: "100%" }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: "0 0 4px", fontSize: 18, fontWeight: 600, color: "#111827" }}>
          App Credentials
        </h2>
        <p style={{ margin: 0, fontSize: 13, color: "#6b7280" }}>
          Your stored OAuth connections — used automatically in chat.
        </p>
      </div>

      {loading ? (
        <div style={{ display: "flex", justifyContent: "center", padding: "48px 0" }}>
          <Spin />
        </div>
      ) : credentials.length === 0 ? (
        <div style={{
          textAlign: "center", color: "#9ca3af", fontSize: 13,
          padding: "48px 12px", border: "1px dashed #e5e7eb", borderRadius: 10,
        }}>
          <LinkOutlined style={{ fontSize: 28, marginBottom: 12, display: "block", color: "#d1d5db" }} />
          No connections yet.
          <br />
          Go to <strong>Apps</strong> and click <strong>Connect</strong> to authorize an MCP server.
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {credentials.map((cred) => {
            const name = displayName(cred);
            const isRevoking = revoking.has(cred.server_id);
            const exp = expiryLabel(cred.expires_at);
            const connected = relativeTime(cred.connected_at);
            const isExpired = exp === "Expired";

            return (
              <div
                key={cred.server_id}
                style={{
                  display: "flex", alignItems: "center", gap: 12,
                  padding: "14px 16px", border: "1px solid #e5e7eb",
                  borderRadius: 10, background: "#fff",
                }}
              >
                {/* Status dot */}
                <div style={{ flexShrink: 0 }}>
                  <CheckCircleOutlined style={{ fontSize: 20, color: isExpired ? "#d1d5db" : "#52c41a" }} />
                </div>

                {/* Info */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {name}
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 2, flexWrap: "wrap" }}>
                    {connected && (
                      <span style={{ fontSize: 12, color: "#9ca3af" }}>
                        Connected {connected}
                      </span>
                    )}
                    <span style={{
                      fontSize: 11, fontWeight: 600,
                      color: isExpired ? "#ef4444" : "#16a34a",
                      background: isExpired ? "#fef2f2" : "#f0fdf4",
                      borderRadius: 4, padding: "1px 6px",
                    }}>
                      {exp}
                    </span>
                  </div>
                </div>

                {/* Revoke */}
                <button
                  onClick={() => handleRevoke(cred.server_id)}
                  disabled={isRevoking}
                  title="Revoke connection"
                  style={{
                    background: "none", border: "1px solid #e5e7eb",
                    borderRadius: 6, padding: "4px 8px",
                    cursor: isRevoking ? "not-allowed" : "pointer",
                    color: "#9ca3af", display: "flex", alignItems: "center",
                    opacity: isRevoking ? 0.5 : 1, flexShrink: 0,
                  }}
                >
                  {isRevoking ? (
                    <Spin size="small" />
                  ) : (
                    <DeleteOutlined style={{ fontSize: 14 }} />
                  )}
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default MCPCredentialsTab;
