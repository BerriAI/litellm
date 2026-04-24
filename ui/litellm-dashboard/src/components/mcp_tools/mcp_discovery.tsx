import React, { useState, useMemo, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search } from "lucide-react";
import { fetchDiscoverableMCPServers } from "../networking";
import { DiscoverableMCPServer, DiscoverMCPServersResponse } from "./types";
import { mcpLogoImg } from "./create_mcp_server";

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
    name.split("").reduce((acc, ch) => acc + ch.charCodeAt(0), 0) % INITIAL_COLORS.length;
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
    <Dialog open={isVisible} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <div className="flex items-center justify-between pb-4 border-b border-border">
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
              <DialogTitle className="text-xl font-semibold">Add MCP Server</DialogTitle>
            </div>
            <Button
              type="button"
              variant="link"
              onClick={onCustomServer}
              className="text-sm font-medium"
            >
              + Custom Server
            </Button>
          </div>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto px-1">
          {/* Filter pills */}
          <div className="flex gap-1.5 flex-wrap mb-3">
            {["All", ...categories].map((cat) => {
              const isSelected = selectedCategory === cat;
              return (
                <button
                  key={cat}
                  type="button"
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-3 py-1 rounded text-xs leading-5 transition-colors ${
                    isSelected
                      ? "bg-foreground text-background border border-foreground font-medium"
                      : "bg-background text-muted-foreground border border-border font-normal hover:bg-muted"
                  }`}
                >
                  {cat}
                </button>
              );
            })}
          </div>

          {/* Search */}
          <div className="relative mb-4">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search servers..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>

          {/* Loading skeleton */}
          {loading && (
            <div className="flex flex-col gap-1">
              {Array.from({ length: 8 }).map((_, i) => (
                <div key={i} className="h-9 rounded-md bg-muted" />
              ))}
            </div>
          )}

          {error && (
            <div className="text-center py-8 text-muted-foreground">
              <span>Failed to load servers: {error}</span>
            </div>
          )}

          {!loading && !error && filteredServers.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              <span>
                No servers found.{" "}
                <button
                  type="button"
                  onClick={onCustomServer}
                  className="text-primary cursor-pointer underline"
                >
                  Add a custom server
                </button>
              </span>
            </div>
          )}

          {/* Server list grouped by category */}
          {!loading &&
            !error &&
            Object.entries(groupedServers).map(([category, categoryServers]) => (
              <div key={category} className="mb-4">
                <div className="text-[11px] font-medium text-muted-foreground uppercase tracking-[0.05em] py-1.5 border-b border-border mb-1">
                  {category}
                </div>
                <div className="grid grid-cols-2 gap-x-4">
                  {categoryServers.map((server) => {
                    const avatar = getInitialAvatar(server.title || server.name);
                    return (
                      <button
                        key={server.name}
                        type="button"
                        onClick={() => onSelectServer(server)}
                        className="flex items-center px-2.5 py-2 rounded-md text-left hover:bg-muted transition-colors"
                      >
                        {server.icon_url ? (
                          <img
                            src={server.icon_url}
                            alt={server.title}
                            className="w-5 h-5 object-contain flex-shrink-0 mr-3"
                            onError={(e) => {
                              const target = e.currentTarget;
                              target.style.display = "none";
                              const next = target.nextElementSibling as HTMLElement;
                              if (next) next.style.display = "flex";
                            }}
                          />
                        ) : null}
                        <div
                          className="w-5 h-5 rounded text-white items-center justify-center font-semibold text-[11px] flex-shrink-0 mr-3"
                          style={{
                            backgroundColor: avatar.backgroundColor,
                            display: server.icon_url ? "none" : "flex",
                          }}
                        >
                          {avatar.initial}
                        </div>
                        <span className="text-sm font-normal text-foreground flex-1 overflow-hidden text-ellipsis whitespace-nowrap">
                          {server.title || server.name}
                        </span>
                        <span className="text-muted-foreground text-sm flex-shrink-0 ml-2">&#8250;</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default MCPDiscovery;
