"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Spin, Input, Button, Tag } from "antd";
import { SearchOutlined, ArrowLeftOutlined, RightOutlined, CheckCircleFilled, LinkOutlined } from "@ant-design/icons";
import { fetchMCPServers, listMCPTools, checkMCPUserCredential, deleteMCPUserCredential, proxyBaseUrl } from "../networking";
import { MCPServer } from "../mcp_tools/types";
import { message } from "antd";

interface Props {
  accessToken: string;
  selectedServers: string[];
  onChange: (servers: string[]) => void;
}

const AVATAR_COLORS = [
  "#1677ff", "#52c41a", "#fa8c16", "#eb2f96", "#722ed1",
  "#13c2c2", "#fa541c", "#2f54eb", "#a0d911", "#faad14",
];

function getAvatarColor(name: string): string {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}

function ServerAvatar({ server, size = 38 }: { server: MCPServer; size?: number }) {
  const name = server.server_name ?? server.alias ?? server.server_id;
  const logoUrl = server.mcp_info?.logo_url;
  const color = getAvatarColor(name);
  const radius = size === 64 ? 16 : 10;
  const fontSize = size === 64 ? 28 : 16;

  const [imgError, setImgError] = useState(false);

  if (logoUrl && !imgError) {
    return (
      <img
        src={logoUrl}
        alt={name}
        onError={() => setImgError(true)}
        style={{ width: size, height: size, borderRadius: radius, objectFit: "contain", flexShrink: 0, background: "#f9fafb", border: "1px solid #e5e7eb" }}
      />
    );
  }

  return (
    <div style={{
      width: size, height: size, borderRadius: radius, background: color,
      display: "flex", alignItems: "center", justifyContent: "center",
      color: "#fff", fontWeight: 700, fontSize, flexShrink: 0,
    }}>
      {name.charAt(0).toUpperCase()}
    </div>
  );
}

type TabKey = "all" | "connected";

const MCPAppsPanel: React.FC<Props> = ({ accessToken, selectedServers, onChange }) => {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [activeTab, setActiveTab] = useState<TabKey>("all");
  const [togglingOn, setTogglingOn] = useState<Set<string>>(new Set());
  const [detailServer, setDetailServer] = useState<MCPServer | null>(null);
  // credential state for the detail view (OAuth2 / BYOK servers)
  const [hasCredential, setHasCredential] = useState<boolean | null>(null);
  const [credLoading, setCredLoading] = useState(false);
  const [disconnecting, setDisconnecting] = useState(false);

  const loadServers = useCallback(() => {
    let cancelled = false;
    setLoading(true);
    fetchMCPServers(accessToken)
      .then((data) => {
        if (cancelled) return;
        const list: MCPServer[] = Array.isArray(data) ? data : (data?.data ?? []);
        setServers(list);
      })
      .catch(() => { if (!cancelled) setServers([]); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [accessToken]);

  useEffect(() => {
    const cancel = loadServers();
    return cancel;
  }, [loadServers]);

  // Handle return from OAuth — URL param ?mcp_oauth_complete=<server_id>
  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const completedServerId = params.get("mcp_oauth_complete");
    if (!completedServerId) return;
    // Remove the param from URL without reload
    const newUrl = new URL(window.location.href);
    newUrl.searchParams.delete("mcp_oauth_complete");
    window.history.replaceState({}, "", newUrl.toString());
    // Reload servers and open the detail for the completed server
    loadServers();
  }, [loadServers]);

  const isOAuth2Server = (s: MCPServer) => (s as MCPServer & { auth_type?: string }).auth_type === "oauth2";

  // When the detail server changes, check credential status for OAuth2 / BYOK servers
  useEffect(() => {
    if (!detailServer) { setHasCredential(null); return; }
    const needsCheck = isOAuth2Server(detailServer) || (detailServer as MCPServer & { is_byok?: boolean }).is_byok;
    if (!needsCheck) { setHasCredential(null); return; }

    // Try from the server list first (has_user_credential is annotated server-side)
    const annotated = (detailServer as MCPServer & { has_user_credential?: boolean }).has_user_credential;
    if (annotated !== undefined) {
      setHasCredential(annotated);
      return;
    }
    // Fall back to individual API call
    setCredLoading(true);
    checkMCPUserCredential(accessToken, detailServer.server_id)
      .then((res) => setHasCredential(res.has_credential))
      .catch(() => setHasCredential(false))
      .finally(() => setCredLoading(false));
  }, [detailServer, accessToken]);

  const handleToggle = async (serverName: string, checked: boolean) => {
    if (!checked) {
      onChange(selectedServers.filter((s) => s !== serverName));
      return;
    }
    setTogglingOn((prev) => new Set(prev).add(serverName));
    try {
      const result = await listMCPTools(accessToken, serverName);
      if (result?.error) {
        message.warning(`Could not load tools for ${serverName}`);
        return;
      }
      onChange([...selectedServers, serverName]);
    } catch {
      message.warning(`Could not load tools for ${serverName}`);
    } finally {
      setTogglingOn((prev) => {
        const next = new Set(prev); next.delete(serverName); return next;
      });
    }
  };

  const handleOAuthSignIn = (server: MCPServer) => {
    const base = proxyBaseUrl ?? "";
    const redirectUri = window.location.href.split("?")[0] + `?mcp_oauth_complete=${server.server_id}`;
    const authorizeUrl = `${base}/v1/mcp/oauth/authorize?server_id=${encodeURIComponent(server.server_id)}&redirect_uri=${encodeURIComponent(redirectUri)}`;
    window.location.href = authorizeUrl;
  };

  const handleDisconnect = async (server: MCPServer) => {
    const name = server.server_name ?? server.alias ?? server.server_id;
    setDisconnecting(true);
    try {
      await deleteMCPUserCredential(accessToken, server.server_id);
      setHasCredential(false);
      onChange(selectedServers.filter((s) => s !== name));
      // Update the server in the list
      setServers((prev) => prev.map((s) =>
        s.server_id === server.server_id
          ? { ...s, has_user_credential: false } as MCPServer
          : s
      ));
    } catch {
      message.error("Failed to disconnect");
    } finally {
      setDisconnecting(false);
    }
  };

  const nameOf = (s: MCPServer) => s.server_name ?? s.alias ?? s.server_id;

  const filtered = servers.filter((s) => {
    const name = nameOf(s);
    const matchesQuery = !query.trim() ||
      name.toLowerCase().includes(query.toLowerCase()) ||
      (s.description ?? "").toLowerCase().includes(query.toLowerCase());
    const matchesTab = activeTab === "all" || selectedServers.includes(name) ||
      (isOAuth2Server(s) && (s as MCPServer & { has_user_credential?: boolean }).has_user_credential);
    return matchesQuery && matchesTab;
  });

  const connectedCount = servers.filter((s) => {
    const name = nameOf(s);
    if (selectedServers.includes(name)) return true;
    if (isOAuth2Server(s) && (s as MCPServer & { has_user_credential?: boolean }).has_user_credential) return true;
    return false;
  }).length;

  // ── Detail view ──
  if (detailServer) {
    const name = nameOf(detailServer);
    const isConnected = selectedServers.includes(name);
    const isTogglingOn = togglingOn.has(name);
    const isOAuth2 = isOAuth2Server(detailServer);
    const isByok = (detailServer as MCPServer & { is_byok?: boolean }).is_byok;
    const needsUserAuth = isOAuth2 || isByok;

    return (
      <div style={{ width: "100%" }}>
        {/* Back */}
        <button
          onClick={() => setDetailServer(null)}
          style={{
            display: "flex", alignItems: "center", gap: 6,
            background: "none", border: "none", cursor: "pointer",
            color: "#6b7280", fontSize: 13, padding: "0 0 20px 0",
          }}
        >
          <ArrowLeftOutlined style={{ fontSize: 12 }} />
          Back
        </button>

        {/* Avatar + name + connect */}
        <div style={{ display: "flex", alignItems: "flex-start", gap: 20, marginBottom: 28 }}>
          <ServerAvatar server={detailServer} size={64} />
          <div style={{ flex: 1 }}>
            <h2 style={{ margin: "0 0 4px", fontSize: 22, fontWeight: 700, color: "#111827" }}>{name}</h2>
            <p style={{ margin: 0, fontSize: 14, color: "#6b7280" }}>{detailServer.description ?? "MCP server"}</p>
          </div>

          {/* Connect button — OAuth2 path */}
          {needsUserAuth && (
            credLoading ? (
              <Spin size="small" />
            ) : hasCredential ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 6 }}>
                <Tag icon={<CheckCircleFilled />} color="success" style={{ margin: 0, fontSize: 12, padding: "4px 10px" }}>
                  Connected
                </Tag>
                <Button
                  size="small"
                  danger
                  loading={disconnecting}
                  onClick={() => handleDisconnect(detailServer)}
                  style={{ borderRadius: 6, fontSize: 12 }}
                >
                  Disconnect
                </Button>
              </div>
            ) : (
              <Button
                type="primary"
                icon={<LinkOutlined />}
                onClick={() => handleOAuthSignIn(detailServer)}
                style={{ borderRadius: 8, fontWeight: 600, height: 38, minWidth: 110 }}
              >
                Sign In
              </Button>
            )
          )}

          {/* Connect button — standard (non-OAuth2) path */}
          {!needsUserAuth && (
            <Button
              type={isConnected ? "default" : "primary"}
              loading={isTogglingOn}
              onClick={() => handleToggle(name, !isConnected)}
              style={{ borderRadius: 8, fontWeight: 600, height: 38, minWidth: 110 }}
            >
              {isConnected ? "Disconnect" : "Connect"}
            </Button>
          )}
        </div>

        {/* Info table */}
        <h3 style={{ margin: "0 0 12px", fontSize: 15, fontWeight: 600, color: "#111827" }}>Information</h3>
        <div style={{ border: "1px solid #e5e7eb", borderRadius: 10, overflow: "hidden" }}>
          {([
            ["Server ID", detailServer.server_id],
            ["Auth", isOAuth2 ? "OAuth 2.0 (sign in with your account)" : isByok ? "API Key (BYOK)" : "None"],
            ["Status", (needsUserAuth ? hasCredential : isConnected) ? "Connected" : "Not connected"],
          ] as [string, string][]).filter(([, v]) => v).map(([label, value], i, arr) => (
            <div key={label} style={{
              display: "flex", padding: "12px 16px",
              borderBottom: i < arr.length - 1 ? "1px solid #f3f4f6" : "none", fontSize: 13,
            }}>
              <span style={{ width: 140, color: "#9ca3af", flexShrink: 0 }}>{label}</span>
              <span style={{ color: "#111827", fontWeight: 500 }}>{value}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ── List view ──
  return (
    <div style={{ width: "100%" }}>

      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20, gap: 16, flexWrap: "wrap" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
            <h2 style={{ margin: 0, fontSize: 18, fontWeight: 600, color: "#111827" }}>Apps</h2>
            <span style={{
              fontSize: 10, fontWeight: 600, color: "#1677ff",
              background: "#e8f4ff", borderRadius: 4, padding: "1px 6px",
              letterSpacing: "0.05em", textTransform: "uppercase",
            }}>Beta</span>
          </div>
          <p style={{ margin: 0, fontSize: 13, color: "#6b7280" }}>
            Connect apps to your chat.
          </p>
        </div>
        <Input
          prefix={<SearchOutlined style={{ color: "#9ca3af", fontSize: 13 }} />}
          placeholder="Search apps..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          allowClear
          style={{ width: 220, borderRadius: 8, fontSize: 13 }}
          size="middle"
        />
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", borderBottom: "1px solid #e5e7eb", marginBottom: 16 }}>
        {(["all", "connected"] as TabKey[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: "8px 16px", border: "none",
              borderBottom: activeTab === tab ? "2px solid #1677ff" : "2px solid transparent",
              cursor: "pointer", fontSize: 13,
              fontWeight: activeTab === tab ? 600 : 400,
              background: "transparent",
              color: activeTab === tab ? "#1677ff" : "#6b7280",
              marginBottom: -1,
            }}
          >
            {tab === "all" ? "All" : `Connected${connectedCount > 0 ? ` (${connectedCount})` : ""}`}
          </button>
        ))}
      </div>

      {/* Grid */}
      {loading ? (
        <div style={{ display: "flex", justifyContent: "center", padding: "48px 0" }}>
          <Spin />
        </div>
      ) : filtered.length === 0 ? (
        <div style={{ textAlign: "center", color: "#9ca3af", fontSize: 13, padding: "48px 12px" }}>
          {servers.length === 0
            ? "No apps configured. Add servers in Tools → MCP Servers."
            : activeTab === "connected" ? "No apps connected yet." : "No apps match your search."}
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 0, border: "1px solid #e5e7eb", borderRadius: 10, overflow: "hidden" }}>
          {filtered.map((server, idx) => {
            const name = nameOf(server);
            const isConnected = selectedServers.includes(name);
            const isOAuth2 = isOAuth2Server(server);
            const hasCred = (server as MCPServer & { has_user_credential?: boolean }).has_user_credential;
            const isLinked = isConnected || (isOAuth2 && hasCred);
            const isLeftCol = idx % 2 === 0;

            return (
              <div
                key={server.server_id}
                onClick={() => setDetailServer(server)}
                style={{
                  display: "flex", alignItems: "center", gap: 12,
                  padding: "14px 16px", background: "#fff",
                  borderRight: isLeftCol ? "1px solid #f3f4f6" : "none",
                  borderBottom: Math.floor(idx / 2) < Math.floor((filtered.length - 1) / 2) ? "1px solid #f3f4f6" : "none",
                  cursor: "pointer", minWidth: 0, transition: "background 0.1s",
                }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.background = "#fafafa"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.background = "#fff"; }}
              >
                <ServerAvatar server={server} size={38} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 14, fontWeight: 500, color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {name}
                  </div>
                  <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {server.description ?? "MCP server"}
                  </div>
                </div>
                {isLinked && (
                  <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#52c41a", flexShrink: 0 }} />
                )}
                {isOAuth2 && !hasCred && (
                  <span style={{ fontSize: 11, color: "#9ca3af", flexShrink: 0, marginRight: 2 }}>Sign In</span>
                )}
                <RightOutlined style={{ fontSize: 11, color: "#d1d5db", flexShrink: 0 }} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default MCPAppsPanel;
