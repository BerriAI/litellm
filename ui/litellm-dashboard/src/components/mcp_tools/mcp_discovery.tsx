import React, { useState, useMemo, useEffect } from "react";
import { Modal, Input } from "antd";
import { Text } from "@tremor/react";
import { fetchDiscoverableMCPServers } from "../networking";
import { DiscoverableMCPServer, DiscoverMCPServersResponse } from "./types";

const { Search } = Input;

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
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ fontSize: 14, fontWeight: 600, color: "#111" }}>
            Add MCP Server
          </span>
          <button
            onClick={onCustomServer}
            style={{
              fontSize: 12,
              padding: "3px 10px",
              borderRadius: 4,
              border: "1px solid #d1d5db",
              background: "#fff",
              color: "#374151",
              cursor: "pointer",
              fontWeight: 500,
            }}
          >
            + Custom Server
          </button>
        </div>
      }
      open={isVisible}
      onCancel={onClose}
      footer={null}
      width={640}
      styles={{
        body: { maxHeight: "70vh", overflowY: "auto", padding: "12px 24px 24px" },
        header: { padding: "16px 24px 0", border: "none" },
      }}
    >
      {/* Filter pills */}
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 8 }}>
        {["All", ...categories].map((cat) => {
          const isSelected = selectedCategory === cat;
          return (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              style={{
                padding: "2px 8px",
                borderRadius: 4,
                border: isSelected ? "1px solid #111827" : "1px solid #e5e7eb",
                background: isSelected ? "#111827" : "#fff",
                color: isSelected ? "#fff" : "#4b5563",
                cursor: "pointer",
                fontSize: 11,
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
        style={{ marginBottom: 12 }}
        size="small"
        allowClear
      />

      {/* Loading skeleton */}
      {loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {Array.from({ length: 8 }).map((_, i) => (
            <div
              key={i}
              style={{
                height: 32,
                borderRadius: 4,
                background: "#f9fafb",
              }}
            />
          ))}
        </div>
      )}

      {error && (
        <div style={{ textAlign: "center", padding: "24px 0", color: "#9ca3af" }}>
          <Text>Failed to load servers: {error}</Text>
        </div>
      )}

      {!loading && !error && filteredServers.length === 0 && (
        <div style={{ textAlign: "center", padding: "24px 0", color: "#9ca3af" }}>
          <Text>
            No servers found.{" "}
            <a
              onClick={onCustomServer}
              style={{ color: "#2563eb", cursor: "pointer", fontSize: 13 }}
            >
              Add a custom server
            </a>
          </Text>
        </div>
      )}

      {/* Server list grouped by category */}
      {!loading &&
        !error &&
        Object.entries(groupedServers).map(([category, categoryServers]) => (
          <div key={category} style={{ marginBottom: 12 }}>
            <div
              style={{
                fontSize: 11,
                fontWeight: 500,
                color: "#9ca3af",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                padding: "4px 0",
                borderBottom: "1px solid #f3f4f6",
                marginBottom: 2,
              }}
            >
              {category}
            </div>
            {categoryServers.map((server) => {
              const avatar = getInitialAvatar(server.title || server.name);
              return (
                <div
                  key={server.name}
                  onClick={() => onSelectServer(server)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    padding: "6px 8px",
                    borderRadius: 4,
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
                        width: 18,
                        height: 18,
                        objectFit: "contain",
                        flexShrink: 0,
                        marginRight: 10,
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
                      width: 18,
                      height: 18,
                      borderRadius: 3,
                      backgroundColor: avatar.backgroundColor,
                      color: "#fff",
                      display: server.icon_url ? "none" : "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontWeight: 600,
                      fontSize: 10,
                      flexShrink: 0,
                      marginRight: 10,
                    }}
                  >
                    {avatar.initial}
                  </div>
                  <span
                    style={{
                      fontSize: 13,
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
                  <span style={{ color: "#d1d5db", fontSize: 12, flexShrink: 0 }}>
                    &#8250;
                  </span>
                </div>
              );
            })}
          </div>
        ))}
    </Modal>
  );
};

export default MCPDiscovery;
