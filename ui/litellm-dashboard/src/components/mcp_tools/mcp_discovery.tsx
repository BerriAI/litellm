import React, { useState, useMemo, useEffect } from "react";
import { Modal, Input, Typography } from "antd";
import { fetchDiscoverableMCPServers } from "../networking";
import { DiscoverableMCPServer, DiscoverMCPServersResponse } from "./types";
import { mcpLogoImg } from "./create_mcp_server";

const { Search } = Input;
const { Text } = Typography;

interface MCPDiscoveryProps {
  isVisible: boolean;
  onClose: () => void;
  onSelectServer: (server: DiscoverableMCPServer) => void;
  onCustomServer: () => void;
  accessToken: string | null;
}

const INITIAL_COLORS = [
  "#3B82F6",
  "#10B981",
  "#F59E0B",
  "#EF4444",
  "#8B5CF6",
  "#EC4899",
  "#06B6D4",
  "#84CC16",
];

function getInitialAvatar(name: string) {
  const initial = name.charAt(0).toUpperCase();
  const colorIndex =
    name.split("").reduce((acc, ch) => acc + ch.charCodeAt(0), 0) %
    INITIAL_COLORS.length;
  return { initial, backgroundColor: INITIAL_COLORS[colorIndex] };
}

const MCPDiscovery: React.FC<MCPDiscoveryProps> = ({
  isVisible,
  onClose,
  onSelectServer,
  onCustomServer,
  accessToken,
}) => {
  const [servers, setServers] = useState<DiscoverableMCPServer[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All");

  useEffect(() => {
    if (isVisible && accessToken) {
      setLoading(true);
      setError(null);
      fetchDiscoverableMCPServers(accessToken)
        .then((data: DiscoverMCPServersResponse) => {
          setServers(data.servers || []);
          setCategories(data.categories || []);
        })
        .catch((err: Error) => {
          setError(err.message || "Failed to load MCP servers");
        })
        .finally(() => {
          setLoading(false);
        });
    }
  }, [isVisible, accessToken]);

  useEffect(() => {
    if (isVisible) {
      setSearchQuery("");
      setSelectedCategory("All");
    }
  }, [isVisible]);

  const filteredServers = useMemo(() => {
    let result = servers;
    if (selectedCategory !== "All") {
      result = result.filter((s) => s.category === selectedCategory);
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (s) =>
          s.name.toLowerCase().includes(q) ||
          s.title.toLowerCase().includes(q) ||
          s.description.toLowerCase().includes(q),
      );
    }
    return result;
  }, [servers, selectedCategory, searchQuery]);

  const groupedServers = useMemo(() => {
    const groups: Record<string, DiscoverableMCPServer[]> = {};
    for (const server of filteredServers) {
      const cat = server.category || "Other";
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(server);
    }
    return groups;
  }, [filteredServers]);

  return (
    <Modal
      title={
        <div className="flex items-center justify-between pb-4 border-b border-gray-100">
          <div className="flex items-center space-x-3">
            <img
              src={mcpLogoImg}
              alt="MCP Logo"
              className="w-8 h-8 object-contain"
              style={{
                height: "20px",
                width: "20px",
                marginRight: "8px",
                objectFit: "contain",
              }}
            />
            <h2 className="text-xl font-semibold text-gray-900">Add MCP Server</h2>
          </div>
          <button
            onClick={onCustomServer}
            className="text-sm text-blue-600 hover:text-blue-800 cursor-pointer bg-transparent border-none font-medium"
          >
            + Custom Server
          </button>
        </div>
      }
      open={isVisible}
      onCancel={onClose}
      footer={null}
      width={1000}
      className="top-8"
      styles={{
        body: { padding: "24px", maxHeight: "70vh", overflowY: "auto" },
        header: { padding: "24px 24px 0 24px", border: "none" },
      }}
    >
      {/* Filter pills */}
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 12 }}>
        {["All", ...categories].map((cat) => {
          const isSelected = selectedCategory === cat;
          return (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              style={{
                padding: "4px 12px",
                borderRadius: 4,
                border: isSelected ? "1px solid #111827" : "1px solid #e5e7eb",
                background: isSelected ? "#111827" : "#fff",
                color: isSelected ? "#fff" : "#4b5563",
                cursor: "pointer",
                fontSize: 12,
                fontWeight: isSelected ? 500 : 400,
                lineHeight: "20px",
              }}
            >
              {cat}
            </button>
          );
        })}
      </div>

      {/* Search */}
      <Search
        placeholder="Search servers..."
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        style={{ marginBottom: 16 }}
        allowClear
      />

      {/* Loading skeleton */}
      {loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {Array.from({ length: 8 }).map((_, i) => (
            <div
              key={i}
              style={{
                height: 36,
                borderRadius: 6,
                background: "#f9fafb",
              }}
            />
          ))}
        </div>
      )}

      {error && (
        <div style={{ textAlign: "center", padding: "32px 0", color: "#9ca3af" }}>
          <Text>Failed to load servers: {error}</Text>
        </div>
      )}

      {!loading && !error && filteredServers.length === 0 && (
        <div style={{ textAlign: "center", padding: "32px 0", color: "#9ca3af" }}>
          <Text>
            No servers found.{" "}
            <a
              onClick={onCustomServer}
              style={{ color: "#2563eb", cursor: "pointer" }}
            >
              Add a custom server
            </a>
          </Text>
        </div>
      )}

      {/* Server list grouped by category — 2 columns */}
      {!loading &&
        !error &&
        Object.entries(groupedServers).map(([category, categoryServers]) => (
          <div key={category} style={{ marginBottom: 16 }}>
            <div
              style={{
                fontSize: 11,
                fontWeight: 500,
                color: "#9ca3af",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                padding: "6px 0",
                borderBottom: "1px solid #f3f4f6",
                marginBottom: 4,
              }}
            >
              {category}
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "0 16px",
              }}
            >
              {categoryServers.map((server) => {
                const avatar = getInitialAvatar(server.title || server.name);
                return (
                  <div
                    key={server.name}
                    onClick={() => onSelectServer(server)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      padding: "8px 10px",
                      borderRadius: 6,
                      cursor: "pointer",
                      transition: "background 0.1s ease",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = "#f9fafb";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = "transparent";
                    }}
                  >
                    {server.icon_url ? (
                      <img
                        src={server.icon_url}
                        alt={server.title}
                        style={{
                          width: 20,
                          height: 20,
                          objectFit: "contain",
                          flexShrink: 0,
                          marginRight: 12,
                        }}
                        onError={(e) => {
                          const target = e.currentTarget;
                          target.style.display = "none";
                          const next = target.nextElementSibling as HTMLElement;
                          if (next) next.style.display = "flex";
                        }}
                      />
                    ) : null}
                    <div
                      style={{
                        width: 20,
                        height: 20,
                        borderRadius: 4,
                        backgroundColor: avatar.backgroundColor,
                        color: "#fff",
                        display: server.icon_url ? "none" : "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontWeight: 600,
                        fontSize: 11,
                        flexShrink: 0,
                        marginRight: 12,
                      }}
                    >
                      {avatar.initial}
                    </div>
                    <span
                      style={{
                        fontSize: 14,
                        fontWeight: 400,
                        color: "#111827",
                        flex: 1,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {server.title || server.name}
                    </span>
                    <span style={{ color: "#d1d5db", fontSize: 14, flexShrink: 0, marginLeft: 8 }}>
                      &#8250;
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
    </Modal>
  );
};

export default MCPDiscovery;
