"use client";

import React, { useEffect, useState } from "react";
import { Spin, Input, Button, Skeleton } from "antd";
import { SearchOutlined, ArrowLeftOutlined, RightOutlined, ToolOutlined } from "@ant-design/icons";
import { fetchMCPServers, listMCPTools } from "../networking";
import { MCPServer, MCPTool, handleTransport } from "../mcp_tools/types";
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
  const [detailServer, setDetailServer] = useState<MCPServer | null>(null);
  const [detailTools, setDetailTools] = useState<MCPTool[]>([]);
  const [loadingTools, setLoadingTools] = useState(false);
  // tool counts per server name, preloaded in background
  const [toolCounts, setToolCounts] = useState<Record<string, number>>({});
  const [loadingCounts, setLoadingCounts] = useState(false);

  const nameOf = (s: MCPServer) => s.server_name ?? s.alias ?? s.server_id;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    // 1. Load servers first — show the list immediately
    fetchMCPServers(accessToken)
      .then((serverData) => {
        if (cancelled) return;
        const list: MCPServer[] = Array.isArray(serverData) ? serverData : (serverData?.data ?? []);
        setServers(list);
        setLoading(false);

        // 2. Fetch tools per server in parallel — each resolves independently and updates counts one by one
        setLoadingCounts(true);
        let remaining = list.length;
        if (remaining === 0) { setLoadingCounts(false); return; }
        list.forEach((s) => {
          listMCPTools(accessToken, s.server_id)
            .then((toolsData) => {
              if (cancelled) return;
              const tools: MCPTool[] = Array.isArray(toolsData?.tools) ? toolsData.tools : [];
              const sname = nameOf(s);
              setToolCounts((prev) => ({ ...prev, [sname]: tools.length }));
            })
            .catch(() => {})
            .finally(() => {
              if (cancelled) return;
              remaining -= 1;
              if (remaining === 0) setLoadingCounts(false);
            });
        });
      })
      .catch(() => {
        if (!cancelled) {
          setServers([]);
          setLoading(false);
        }
      });
    return () => { cancelled = true; };
  }, [accessToken]);

  const handleToggle = async (serverName: string, checked: boolean, serverId?: string) => {
    if (!checked) {
      onChange(selectedServers.filter((s) => s !== serverName));
      return;
    }
    setTogglingOn((prev) => new Set(prev).add(serverName));
    try {
      // Use UUID if available, fall back to name (for connectivity check only)
      const idToFetch = serverId ?? serverName;
      const result = await listMCPTools(accessToken, idToFetch);
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

  // Fetch tools for the detail view — server_id must be the UUID
  useEffect(() => {
    if (!detailServer) {
      setDetailTools([]);
      return;
    }
    let cancelled = false;
    setLoadingTools(true);
    listMCPTools(accessToken, detailServer.server_id)
      .then((result) => {
        if (cancelled) return;
        // API returns { tools: [...], error: null }
        const tools: MCPTool[] = Array.isArray(result?.tools) ? result.tools : [];
        setDetailTools(tools);
      })
      .catch(() => {
        if (!cancelled) setDetailTools([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingTools(false);
      });
    return () => { cancelled = true; };
  }, [detailServer, accessToken]);

  const filtered = servers.filter((s) => {
    const name = nameOf(s);
    const matchesQuery = !query.trim() ||
      name.toLowerCase().includes(query.toLowerCase()) ||
      (s.description ?? "").toLowerCase().includes(query.toLowerCase());
    const matchesTab = activeTab === "all" || selectedServers.includes(name);
    return matchesQuery && matchesTab;
  });

  const connectedCount = servers.filter((s) => selectedServers.includes(nameOf(s))).length;

  // Total tools available across all servers (based on preloaded counts)
  const totalTools = Object.values(toolCounts).reduce((sum, n) => sum + n, 0);

  // ── Detail view ──
  if (detailServer) {
    const name = nameOf(detailServer);
    const isConnected = selectedServers.includes(name);
    const isTogglingOn = togglingOn.has(name);
    const color = getAvatarColor(name);

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
          <div style={{
            width: 64, height: 64, borderRadius: 16,
            background: color, display: "flex",
            alignItems: "center", justifyContent: "center",
            color: "#fff", fontWeight: 700, fontSize: 28, flexShrink: 0,
          }}>
            {name.charAt(0).toUpperCase()}
          </div>
          <div style={{ flex: 1 }}>
            <h2 style={{ margin: "0 0 4px", fontSize: 22, fontWeight: 700, color: "#111827" }}>{name}</h2>
            <p style={{ margin: 0, fontSize: 14, color: "#6b7280" }}>{detailServer.description ?? "MCP server"}</p>
          </div>
          <Button
            type={isConnected ? "default" : "primary"}
            loading={isTogglingOn}
            onClick={() => handleToggle(name, !isConnected, detailServer.server_id)}
            style={{ borderRadius: 8, fontWeight: 600, height: 38, minWidth: 110 }}
          >
            {isConnected ? "Disconnect" : "Connect"}
          </Button>
        </div>

        {/* Info table */}
        <h3 style={{ margin: "0 0 12px", fontSize: 15, fontWeight: 600, color: "#111827" }}>Information</h3>
        <div style={{ border: "1px solid #e5e7eb", borderRadius: 10, overflow: "hidden", marginBottom: 28 }}>
          {[
            ["Server ID", detailServer.server_id],
            ["Transport", handleTransport(detailServer.transport, detailServer.spec_path)],
            ["Status", isConnected ? "Connected" : "Not connected"],
          ].filter(([, v]) => v).map(([label, value], i, arr) => (
            <div key={label} style={{
              display: "flex",
              padding: "12px 16px",
              borderBottom: i < arr.length - 1 ? "1px solid #f3f4f6" : "none",
              fontSize: 13,
            }}>
              <span style={{ width: 140, color: "#9ca3af", flexShrink: 0 }}>{label}</span>
              <span style={{ color: "#111827", fontWeight: 500 }}>{value}</span>
            </div>
          ))}
        </div>

        {/* Tools section */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
          <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600, color: "#111827" }}>Available Tools</h3>
          {!loadingTools && (
            <span style={{
              fontSize: 11, fontWeight: 600, color: "#6b7280",
              background: "#f3f4f6", borderRadius: 4, padding: "1px 6px",
            }}>{detailTools.length}</span>
          )}
        </div>
        {loadingTools ? (
          <div style={{ display: "flex", justifyContent: "center", padding: "24px 0" }}>
            <Spin size="small" />
          </div>
        ) : detailTools.length === 0 ? (
          <div style={{ color: "#9ca3af", fontSize: 13, padding: "8px 0" }}>
            No tools available
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {detailTools.map((tool) => (
              <div key={tool.name} style={{
                border: "1px solid #e5e7eb", borderRadius: 8,
                padding: "10px 14px", background: "#fafafa",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: tool.description ? 4 : 0 }}>
                  <ToolOutlined style={{ fontSize: 13, color: "#6b7280" }} />
                  <span style={{ fontSize: 13, fontWeight: 600, color: "#111827", fontFamily: "monospace" }}>{tool.name}</span>
                </div>
                {tool.description && (
                  <p style={{ margin: 0, fontSize: 12, color: "#6b7280", paddingLeft: 21 }}>{tool.description}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  // ── List view ──
  return (
    <div style={{ width: "100%" }}>

      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20, gap: 16, flexWrap: "wrap" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <h2 style={{ margin: 0, fontSize: 18, fontWeight: 600, color: "#111827" }}>MCP Servers</h2>
            <span style={{
              fontSize: 10, fontWeight: 600, color: "#1677ff",
              background: "#e8f4ff", borderRadius: 4, padding: "1px 6px",
              letterSpacing: "0.05em", textTransform: "uppercase",
            }}>Beta</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <p style={{ margin: 0, fontSize: 13, color: "#6b7280" }}>
              Connect tools to your chat.
            </p>
            {loadingCounts ? (
              <span style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: "#9ca3af" }}>
                <Spin size="small" style={{ transform: "scale(0.7)" }} />
                Loading tools...
              </span>
            ) : totalTools > 0 ? (
              <span style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12, color: "#6b7280" }}>
                <ToolOutlined style={{ fontSize: 11 }} />
                {totalTools} tool{totalTools !== 1 ? "s" : ""} available
              </span>
            ) : null}
          </div>
        </div>
        <Input
          prefix={<SearchOutlined style={{ color: "#9ca3af", fontSize: 13 }} />}
          placeholder="Search servers..."
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
            ? "No MCP servers configured. Add servers in Tools → MCP Servers."
            : activeTab === "connected" ? "No servers connected yet." : "No servers match your search."}
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 0, border: "1px solid #e5e7eb", borderRadius: 10, overflow: "hidden" }}>
          {filtered.map((server, idx) => {
            const name = nameOf(server);
            const isConnected = selectedServers.includes(name);
            const color = getAvatarColor(name);
            const isLeftCol = idx % 2 === 0;
            const count = toolCounts[name];

            return (
              <div
                key={server.server_id}
                onClick={() => setDetailServer(server)}
                style={{
                  display: "flex", alignItems: "center", gap: 12,
                  padding: "14px 16px", background: "#fff",
                  borderRight: isLeftCol ? "1px solid #f3f4f6" : "none",
                  borderBottom: Math.floor(idx / 2) < Math.floor((filtered.length - 1) / 2) ? "1px solid #f3f4f6" : "none",
                  cursor: "pointer", minWidth: 0,
                  transition: "background 0.1s",
                }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.background = "#fafafa"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.background = "#fff"; }}
              >
                <div style={{
                  width: 38, height: 38, borderRadius: 10, background: color,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  color: "#fff", fontWeight: 700, fontSize: 16, flexShrink: 0,
                }}>
                  {name.charAt(0).toUpperCase()}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 14, fontWeight: 500, color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {name}
                  </div>
                  <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 1, display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {server.description ?? "MCP server"}
                    </span>
                    {count !== undefined ? (
                      count > 0 ? (
                        <span style={{ flexShrink: 0, display: "flex", alignItems: "center", gap: 3, color: "#9ca3af" }}>
                          · <ToolOutlined style={{ fontSize: 10 }} /> {count}
                        </span>
                      ) : null
                    ) : loadingCounts ? (
                      <Skeleton.Input active size="small" style={{ width: 28, height: 12, minWidth: 28, flexShrink: 0 }} />
                    ) : null}
                  </div>
                </div>
                {isConnected && (
                  <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#1677ff", flexShrink: 0 }} />
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
