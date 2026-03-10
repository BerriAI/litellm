"use client";

import React, { useEffect, useState } from "react";
import { Switch, Spin, Input } from "antd";
import { SearchOutlined, RightOutlined } from "@ant-design/icons";
import { fetchMCPServers, listMCPTools } from "../networking";
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

type TabKey = "all" | "connected";

const MCPAppsPanel: React.FC<Props> = ({ accessToken, selectedServers, onChange }) => {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [activeTab, setActiveTab] = useState<TabKey>("all");
  const [togglingOn, setTogglingOn] = useState<Set<string>>(new Set());
  const [expandedServer, setExpandedServer] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchMCPServers(accessToken)
      .then((data) => {
        if (cancelled) return;
        const list: MCPServer[] = Array.isArray(data) ? data : (data?.data ?? []);
        setServers(list);
      })
      .catch(() => {
        if (!cancelled) setServers([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [accessToken]);

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
        const next = new Set(prev);
        next.delete(serverName);
        return next;
      });
    }
  };

  const nameOf = (s: MCPServer) => s.server_name ?? s.alias ?? s.server_id;

  const filtered = servers.filter((s) => {
    const name = nameOf(s);
    const matchesQuery = !query.trim() ||
      name.toLowerCase().includes(query.toLowerCase()) ||
      (s.description ?? "").toLowerCase().includes(query.toLowerCase());
    const matchesTab = activeTab === "all" || selectedServers.includes(name);
    return matchesQuery && matchesTab;
  });

  const connectedCount = servers.filter((s) => selectedServers.includes(nameOf(s))).length;

  return (
    <div style={{ width: "100%", maxWidth: 800, margin: "0 auto" }}>

      {/* ── Page header ── */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 28, gap: 16, flexWrap: "wrap" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <h1 style={{ margin: 0, fontSize: 26, fontWeight: 700, color: "#111827", letterSpacing: "-0.02em" }}>
              MCP Servers
            </h1>
            <span style={{
              fontSize: 11, fontWeight: 600, color: "#1677ff",
              background: "#e8f4ff", borderRadius: 4, padding: "2px 7px",
              letterSpacing: "0.04em", textTransform: "uppercase",
            }}>
              BETA
            </span>
          </div>
          <p style={{ margin: 0, fontSize: 14, color: "#6b7280" }}>
            Connect tools to your chat. Toggle servers to use them in every message.
          </p>
        </div>

        {/* Search */}
        <div style={{ width: 240, flexShrink: 0 }}>
          <Input
            prefix={<SearchOutlined style={{ color: "#9ca3af", fontSize: 14 }} />}
            placeholder="Search servers..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            allowClear
            style={{ borderRadius: 20, fontSize: 14, height: 38 }}
          />
        </div>
      </div>

      {/* ── Hero banner ── */}
      {!query && (
        <div style={{
          borderRadius: 16,
          background: "linear-gradient(135deg, #1677ff 0%, #36cfc9 60%, #faad14 100%)",
          padding: "28px 32px",
          marginBottom: 24,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 24,
          overflow: "hidden",
          position: "relative",
        }}>
          <div style={{ zIndex: 1 }}>
            <div style={{ fontSize: 20, fontWeight: 700, color: "#fff", marginBottom: 6, letterSpacing: "-0.01em" }}>
              Supercharge your chat with MCP tools
            </div>
            <div style={{ fontSize: 14, color: "rgba(255,255,255,0.85)", maxWidth: 400 }}>
              MCP servers give Claude access to external data, APIs, and actions — right inside your conversation.
            </div>
            {connectedCount > 0 && (
              <div style={{
                marginTop: 14,
                display: "inline-flex", alignItems: "center", gap: 6,
                background: "rgba(255,255,255,0.2)", borderRadius: 20,
                padding: "5px 14px", fontSize: 13, color: "#fff", fontWeight: 500,
              }}>
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#52c41a", flexShrink: 0, display: "inline-block" }} />
                {connectedCount} server{connectedCount > 1 ? "s" : ""} connected
              </div>
            )}
          </div>
          {/* Decorative circles */}
          <div style={{
            position: "absolute", right: -20, top: -20,
            width: 160, height: 160, borderRadius: "50%",
            background: "rgba(255,255,255,0.08)",
          }} />
          <div style={{
            position: "absolute", right: 60, bottom: -40,
            width: 120, height: 120, borderRadius: "50%",
            background: "rgba(255,255,255,0.06)",
          }} />
        </div>
      )}

      {/* ── Tabs ── */}
      <div style={{ display: "flex", gap: 4, marginBottom: 16 }}>
        {(["all", "connected"] as TabKey[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: "6px 16px",
              borderRadius: 20,
              border: "none",
              cursor: "pointer",
              fontSize: 14,
              fontWeight: activeTab === tab ? 600 : 400,
              background: activeTab === tab ? "#111827" : "transparent",
              color: activeTab === tab ? "#fff" : "#6b7280",
              transition: "all 0.15s",
            }}
            onMouseEnter={(e) => {
              if (activeTab !== tab) (e.currentTarget as HTMLButtonElement).style.background = "#f3f4f6";
            }}
            onMouseLeave={(e) => {
              if (activeTab !== tab) (e.currentTarget as HTMLButtonElement).style.background = "transparent";
            }}
          >
            {tab === "all" ? "All" : `Connected${connectedCount > 0 ? ` (${connectedCount})` : ""}`}
          </button>
        ))}
      </div>

      {/* ── Server grid ── */}
      {loading ? (
        <div style={{ display: "flex", justifyContent: "center", padding: "60px 0" }}>
          <Spin size="large" />
        </div>
      ) : filtered.length === 0 ? (
        <div style={{ textAlign: "center", color: "#9ca3af", fontSize: 14, padding: "60px 12px" }}>
          {servers.length === 0
            ? "No MCP servers configured. Add servers in Tools → MCP Servers."
            : activeTab === "connected"
              ? "No servers connected yet. Toggle a server below to connect it."
              : "No servers match your search."}
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, border: "1px solid #e5e7eb", borderRadius: 12, overflow: "hidden" }}>
          {filtered.map((server) => {
            const name = nameOf(server);
            const isConnected = selectedServers.includes(name);
            const isTogglingOn = togglingOn.has(name);
            const isExpanded = expandedServer === name;
            const color = getAvatarColor(name);

            return (
              <div
                key={server.server_id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                  padding: "16px 20px",
                  background: isExpanded ? "#f9fafb" : "#fff",
                  cursor: "pointer",
                  borderBottom: "1px solid #f0f0f0",
                  transition: "background 0.12s",
                }}
                onClick={() => setExpandedServer(isExpanded ? null : name)}
                onMouseEnter={(e) => {
                  if (!isExpanded) (e.currentTarget as HTMLDivElement).style.background = "#fafafa";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLDivElement).style.background = isExpanded ? "#f9fafb" : "#fff";
                }}
              >
                {/* Avatar */}
                <div style={{
                  width: 40, height: 40, borderRadius: 10,
                  background: color, display: "flex",
                  alignItems: "center", justifyContent: "center",
                  color: "#fff", fontWeight: 700, fontSize: 16,
                  flexShrink: 0, boxShadow: "0 1px 4px rgba(0,0,0,0.12)",
                }}>
                  {name.charAt(0).toUpperCase()}
                </div>

                {/* Info */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: 14, fontWeight: 600, color: "#111827",
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {name}
                  </div>
                  <div style={{
                    fontSize: 12, color: "#6b7280", marginTop: 2,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {server.description ?? "MCP server"}
                  </div>
                </div>

                {/* Right side: toggle (when expanded) or chevron */}
                {isExpanded ? (
                  <div onClick={(e) => e.stopPropagation()}>
                    <Switch
                      size="small"
                      checked={isConnected}
                      loading={isTogglingOn}
                      onChange={(checked) => handleToggle(name, checked)}
                    />
                  </div>
                ) : (
                  <RightOutlined style={{ fontSize: 12, color: "#9ca3af", flexShrink: 0 }} />
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default MCPAppsPanel;
