import React, { useState, useMemo, useEffect } from "react";
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/cva.config";
import { fetchDiscoverableMCPServers } from "@/components/networking";
import { DiscoverableMCPServer, DiscoverMCPServersResponse } from "@/components/mcp_tools/types";
import { mcpLogoImg } from "./create_mcp_server";
import { resolveLogoSrc } from "@/lib/assetPaths";

interface MCPDiscoveryProps {
  isVisible: boolean;
  onClose: () => void;
  onSelectServer: (server: DiscoverableMCPServer) => void;
  onCustomServer: () => void;
  accessToken: string | null;
}

const INITIAL_COLORS = [
  "bg-blue-500",
  "bg-emerald-500",
  "bg-amber-500",
  "bg-red-500",
  "bg-violet-500",
  "bg-pink-500",
  "bg-cyan-500",
  "bg-lime-500",
];

function getInitialAvatar(name: string) {
  const initial = name.charAt(0).toUpperCase();
  const colorIndex = name.split("").reduce((acc, ch) => acc + ch.charCodeAt(0), 0) % INITIAL_COLORS.length;
  return { initial, backgroundClass: INITIAL_COLORS[colorIndex] };
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
    <Dialog open={isVisible} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-[1000px]">
        <DialogHeader>
          <div className="flex items-center justify-between border-b border-border pb-4">
            <div className="flex items-center space-x-3">
              <img src={resolveLogoSrc(mcpLogoImg)} alt="MCP Logo" className="mr-2 size-5 object-contain" />
              <DialogTitle className="text-xl font-semibold">Add MCP Server</DialogTitle>
            </div>
            <Button variant="link" size="sm" onClick={onCustomServer}>
              + Custom Server
            </Button>
          </div>
        </DialogHeader>

        <div className="max-h-[70vh] overflow-y-auto">
          {/* Filter pills */}
          <div className="mb-3 flex flex-wrap gap-1.5">
            {["All", ...categories].map((cat) => {
              const isSelected = selectedCategory === cat;
              return (
                <Button
                  key={cat}
                  size="sm"
                  variant={isSelected ? "default" : "outline"}
                  onClick={() => setSelectedCategory(cat)}
                >
                  {cat}
                </Button>
              );
            })}
          </div>

          {/* Search */}
          <InputGroup className="mb-4 w-full">
            <InputGroupAddon>
              <Search className="size-4 text-muted-foreground" />
            </InputGroupAddon>
            <InputGroupInput
              placeholder="Search servers..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </InputGroup>

          {/* Loading skeleton */}
          {loading && (
            <div className="flex flex-col gap-1">
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-9 rounded-md" />
              ))}
            </div>
          )}

          {error && (
            <div className="py-8 text-center text-muted-foreground">
              <p className="text-sm">Failed to load servers: {error}</p>
            </div>
          )}

          {!loading && !error && filteredServers.length === 0 && (
            <div className="py-8 text-center text-muted-foreground">
              <p className="text-sm">
                No servers found.{" "}
                <Button variant="link" size="sm" onClick={onCustomServer}>
                  Add a custom server
                </Button>
              </p>
            </div>
          )}

          {/* Server list grouped by category — 2 columns */}
          {!loading &&
            !error &&
            Object.entries(groupedServers).map(([category, categoryServers]) => (
              <div key={category} className="mb-4">
                <div className="mb-1 border-b border-border py-1.5 text-[11px] font-medium tracking-wider text-muted-foreground uppercase">
                  {category}
                </div>
                <div className="grid grid-cols-2 gap-x-4">
                  {categoryServers.map((server) => {
                    const avatar = getInitialAvatar(server.title || server.name);
                    return (
                      <div
                        key={server.name}
                        onClick={() => onSelectServer(server)}
                        className="flex cursor-pointer items-center rounded-md px-2.5 py-2 transition-colors hover:bg-accent"
                      >
                        {server.icon_url ? (
                          <img
                            src={resolveLogoSrc(server.icon_url)}
                            alt={server.title}
                            className="mr-3 size-5 shrink-0 object-contain"
                            onError={(e) => {
                              const target = e.currentTarget;
                              target.style.display = "none";
                              const next = target.nextElementSibling as HTMLElement;
                              if (next) next.style.display = "flex";
                            }}
                          />
                        ) : null}
                        <div
                          className={cn(
                            "mr-3 size-5 shrink-0 items-center justify-center rounded-sm text-[11px] font-semibold text-white",
                            avatar.backgroundClass,
                            server.icon_url ? "hidden" : "flex",
                          )}
                        >
                          {avatar.initial}
                        </div>
                        <span className="flex-1 truncate text-sm">{server.title || server.name}</span>
                        <span className="ml-2 shrink-0 text-sm text-muted-foreground">&#8250;</span>
                      </div>
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
