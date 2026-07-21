"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search, ArrowLeft, ChevronRight, Wrench, CheckCircle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  deleteMCPOAuthUserCredential,
  fetchMCPServers,
  getMCPOAuthUserCredentialStatus,
  listMCPTools,
} from "../networking";
import { AUTH_TYPE, MCPServer, MCPTool, handleTransport, isAutoConnectedAuthType } from "../mcp_tools/types";
import MessageManager from "@/components/molecules/message_manager";
import { useUserMcpOAuthFlow } from "@/hooks/useUserMcpOAuthFlow";

interface OAuth2ConnectButtonProps {
  server: MCPServer;
  accessToken: string;
  onConnect: (serverId: string) => void;
  variant?: "badge" | "button";
}

const OAuth2ConnectButton: React.FC<OAuth2ConnectButtonProps> = ({
  server,
  accessToken,
  onConnect,
  variant = "badge",
}) => {
  const name = server.server_name ?? server.alias ?? server.server_id;
  const { startOAuthFlow, status } = useUserMcpOAuthFlow({
    accessToken,
    serverId: server.server_id,
    serverAlias: name,
    onSuccess: useCallback(() => onConnect(server.server_id), [onConnect, server.server_id]),
  });

  const loading = status === "authorizing" || status === "exchanging";

  if (variant === "button") {
    return (
      <Button onClick={startOAuthFlow} disabled={loading} className="font-semibold h-[38px] min-w-[110px]">
        {loading && <Loader2 className="h-4 w-4 animate-spin mr-1.5" />}
        {loading ? "Connecting\u2026" : "Connect"}
      </Button>
    );
  }

  return (
    <span
      onClick={(e) => {
        e.stopPropagation();
        if (!loading) startOAuthFlow();
      }}
      className={`text-[11px] font-semibold rounded-md px-2 py-0.5 shrink-0 whitespace-nowrap ${
        loading
          ? "text-muted-foreground bg-muted cursor-default"
          : "text-primary-foreground bg-primary cursor-pointer hover:bg-primary/90"
      }`}
    >
      {loading ? "Connecting\u2026" : "Connect"}
    </span>
  );
};

interface Props {
  accessToken: string;
  selectedServers: string[];
  onChange: (servers: string[]) => void;
}

const AVATAR_COLORS = [
  "#1677ff",
  "#52c41a",
  "#fa8c16",
  "#eb2f96",
  "#722ed1",
  "#13c2c2",
  "#fa541c",
  "#2f54eb",
  "#a0d911",
  "#faad14",
];

function getAvatarColor(name: string): string {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}

type TabKey = "all" | "connected";

const TOOLS_FETCH_CONCURRENCY = 5;

const MCPAppsPanel: React.FC<Props> = ({ accessToken, selectedServers, onChange }) => {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [activeTab, setActiveTab] = useState<TabKey>("all");
  const [togglingOn, setTogglingOn] = useState<Set<string>>(new Set());
  const [detailServer, setDetailServer] = useState<MCPServer | null>(null);
  const [toolCounts, setToolCounts] = useState<Record<string, number>>({});
  const [loadingCounts, setLoadingCounts] = useState(false);
  const [oauthConnected, setOauthConnected] = useState<Set<string>>(new Set());

  const serversRef = useRef<MCPServer[]>([]);
  useEffect(() => {
    serversRef.current = servers;
  }, [servers]);
  const selectedServersRef = useRef<string[]>(selectedServers);
  useEffect(() => {
    selectedServersRef.current = selectedServers;
  }, [selectedServers]);
  const onChangeRef = useRef(onChange);
  useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  const nameOf = (s: MCPServer) => s.server_name ?? s.alias ?? s.server_id;

  const fetchLoadCancelledRef = useRef(false);

  const fetchToolCount = useCallback(
    async (server: MCPServer) => {
      try {
        const toolsData = await listMCPTools(accessToken, server.server_id);
        if (fetchLoadCancelledRef.current) return;
        const tools: MCPTool[] = Array.isArray(toolsData?.tools) ? toolsData.tools : [];
        setToolCounts((prev) => ({ ...prev, [nameOf(server)]: tools.length }));
      } catch {
        // ignore
      }
    },
    [accessToken],
  );

  const checkOauthCredential = useCallback(
    async (server: MCPServer) => {
      try {
        const status = await getMCPOAuthUserCredentialStatus(accessToken, server.server_id);
        if (fetchLoadCancelledRef.current) return;
        if (status.has_credential && !status.is_expired) {
          setOauthConnected((prev) => new Set(prev).add(server.server_id));
        }
      } catch {
        // ignore
      }
    },
    [accessToken],
  );

  useEffect(() => {
    fetchLoadCancelledRef.current = false;

    fetchMCPServers(accessToken)
      .then(async (serverData) => {
        if (fetchLoadCancelledRef.current) return;
        const list: MCPServer[] = Array.isArray(serverData) ? serverData : serverData?.data ?? [];
        setServers(list);
        setLoading(false);

        setLoadingCounts(true);
        const chunks = Array.from({ length: Math.ceil(list.length / TOOLS_FETCH_CONCURRENCY) }, (_, i) =>
          list.slice(i * TOOLS_FETCH_CONCURRENCY, (i + 1) * TOOLS_FETCH_CONCURRENCY),
        );
        for (const chunk of chunks) {
          if (fetchLoadCancelledRef.current) return;
          await Promise.allSettled(chunk.map((s) => fetchToolCount(s)));
        }
        if (!fetchLoadCancelledRef.current) setLoadingCounts(false);

        const oauthServers = list.filter((s) => s.auth_type === AUTH_TYPE.OAUTH2);
        oauthServers.forEach((s) => checkOauthCredential(s));
      })
      .catch(() => {
        if (!fetchLoadCancelledRef.current) {
          setServers([]);
          setLoading(false);
        }
      });
    return () => {
      fetchLoadCancelledRef.current = true;
    };
  }, [accessToken, fetchToolCount, checkOauthCredential]);

  useEffect(() => {
    if (oauthConnected.size === 0) return;
    const namesToAdd = serversRef.current
      .filter((s) => oauthConnected.has(s.server_id) && !selectedServersRef.current.includes(nameOf(s)))
      .map(nameOf);
    if (namesToAdd.length > 0) {
      onChangeRef.current([...selectedServersRef.current, ...namesToAdd]);
    }
  }, [oauthConnected]);

  const handleToggle = async (serverName: string, checked: boolean, serverId?: string) => {
    if (!checked) {
      onChange(selectedServers.filter((s) => s !== serverName));
      if (serverId) {
        setOauthConnected((prev) => {
          const next = new Set(prev);
          next.delete(serverId);
          return next;
        });
      }
      return;
    }
    setTogglingOn((prev) => new Set(prev).add(serverName));
    try {
      const idToFetch = serverId ?? serverName;
      const result = await listMCPTools(accessToken, idToFetch);
      if (result?.error) {
        MessageManager.warning(`Could not load tools for ${serverName}`);
        return;
      }
      if (!selectedServersRef.current.includes(serverName)) {
        onChange([...selectedServersRef.current, serverName]);
      }
    } catch {
      MessageManager.warning(`Could not load tools for ${serverName}`);
    } finally {
      setTogglingOn((prev) => {
        const next = new Set(prev);
        next.delete(serverName);
        return next;
      });
    }
  };

  const { data: detailToolsResult, isLoading: loadingTools } = useQuery({
    queryKey: ["mcp-apps-panel-detail-tools", detailServer?.server_id],
    queryFn: () => listMCPTools(accessToken, detailServer!.server_id),
    enabled: !!detailServer,
  });
  const detailTools: MCPTool[] = Array.isArray(detailToolsResult?.tools) ? detailToolsResult.tools : [];

  const filtered = servers.filter((s) => {
    const name = nameOf(s);
    const matchesQuery =
      !query.trim() ||
      name.toLowerCase().includes(query.toLowerCase()) ||
      (s.description ?? "").toLowerCase().includes(query.toLowerCase());
    const matchesTab = activeTab === "all" || selectedServers.includes(name);
    return matchesQuery && matchesTab;
  });

  const connectedCount = servers.filter((s) => selectedServers.includes(nameOf(s))).length;
  const totalTools = Object.values(toolCounts).reduce((sum, n) => sum + n, 0);

  if (detailServer) {
    const name = nameOf(detailServer);
    const isConnected = selectedServers.includes(name);
    const isTogglingOn = togglingOn.has(name);
    const color = getAvatarColor(name);

    return (
      <div className="w-full">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setDetailServer(null)}
          className="-ml-3 mb-5 gap-1.5 text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3 w-3" />
          Back
        </Button>

        <div className="flex items-start gap-5 mb-7">
          {detailServer.mcp_info?.logo_url ? (
            <img
              src={detailServer.mcp_info.logo_url}
              alt={`${name} logo`}
              className="w-16 h-16 rounded-2xl object-contain shrink-0 bg-muted/50"
              onError={(e) => {
                const el = e.target as HTMLImageElement;
                el.style.display = "none";
                if (el.nextElementSibling) (el.nextElementSibling as HTMLElement).style.display = "flex";
              }}
            />
          ) : null}
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center text-white font-bold text-[28px] shrink-0"
            style={{
              background: color,
              display: detailServer.mcp_info?.logo_url ? "none" : "flex",
            }}
          >
            {name.charAt(0).toUpperCase()}
          </div>
          <div className="flex-1">
            <h2 className="m-0 mb-1 text-[22px] font-bold text-foreground">{name}</h2>
            <p className="m-0 text-sm text-muted-foreground">{detailServer.description ?? "MCP server"}</p>
          </div>
          {isAutoConnectedAuthType(detailServer.auth_type) && (
            <span className="inline-flex items-center gap-1.5 h-[38px] px-3 rounded-md bg-emerald-50 text-emerald-700 text-sm font-semibold">
              <CheckCircle className="h-4 w-4" />
              Connected via your organization sign-in
            </span>
          )}
          {!isAutoConnectedAuthType(detailServer.auth_type) &&
            (detailServer.auth_type === AUTH_TYPE.OAUTH2 ? (
              oauthConnected.has(detailServer.server_id) ? (
                <Button
                  variant="destructive"
                  onClick={async () => {
                    try {
                      await deleteMCPOAuthUserCredential(accessToken, detailServer.server_id);
                    } catch (_) {
                      // Ignore
                    }
                    setOauthConnected((prev) => {
                      const n = new Set(prev);
                      n.delete(detailServer.server_id);
                      return n;
                    });
                    onChangeRef.current(selectedServersRef.current.filter((s) => s !== name));
                  }}
                  className="font-semibold h-[38px] min-w-[110px]"
                >
                  Disconnect
                </Button>
              ) : (
                <OAuth2ConnectButton
                  server={detailServer}
                  accessToken={accessToken}
                  onConnect={(id) => {
                    setOauthConnected((prev) => new Set(prev).add(id));
                  }}
                  variant="button"
                />
              )
            ) : (
              <Button
                variant={isConnected ? "outline" : "default"}
                disabled={isTogglingOn}
                onClick={() => handleToggle(name, !isConnected, detailServer.server_id)}
                className="font-semibold h-[38px] min-w-[110px]"
              >
                {isTogglingOn && <Loader2 className="h-4 w-4 animate-spin mr-1.5" />}
                {isConnected ? "Disconnect" : "Connect"}
              </Button>
            ))}
        </div>

        <h3 className="m-0 mb-3 text-[15px] font-semibold text-foreground">Information</h3>
        <div className="border rounded-lg overflow-hidden mb-7">
          {[
            ["Server ID", detailServer.server_id],
            ["Transport", handleTransport(detailServer.transport, detailServer.spec_path)],
            ["Status", isConnected ? "Connected" : "Not connected"],
          ]
            .filter(([, v]) => v)
            .map(([label, value], i, arr) => (
              <div key={label} className={`flex px-4 py-3 text-[13px] ${i < arr.length - 1 ? "border-b" : ""}`}>
                <span className="w-[140px] text-muted-foreground shrink-0">{label}</span>
                <span className="text-foreground font-medium">{value}</span>
              </div>
            ))}
        </div>

        <div className="flex items-center gap-2 mb-3">
          <h3 className="m-0 text-[15px] font-semibold text-foreground">Available Tools</h3>
          {!loadingTools && (
            <span className="text-[11px] font-semibold text-muted-foreground bg-muted rounded px-1.5 py-0.5">
              {detailTools.length}
            </span>
          )}
        </div>
        {loadingTools ? (
          <div className="flex flex-col gap-2">
            {Array.from({ length: 3 }, (_, i) => (
              <div key={i} className="border rounded-lg px-3.5 py-2.5 bg-muted/30 flex flex-col gap-1.5">
                <Skeleton className="h-3.5 w-1/3" />
                <Skeleton className="h-3 w-2/3" />
              </div>
            ))}
          </div>
        ) : detailTools.length === 0 ? (
          <div className="text-muted-foreground text-[13px] py-2">No tools available</div>
        ) : (
          <div className="flex flex-col gap-2">
            {detailTools.map((tool) => (
              <div key={tool.name} className="border rounded-lg px-3.5 py-2.5 bg-muted/30">
                <div className={`flex items-center gap-2 ${tool.description ? "mb-1" : ""}`}>
                  <Wrench className="h-3.5 w-3.5 text-muted-foreground" />
                  <span className="text-[13px] font-semibold text-foreground font-mono">{tool.name}</span>
                </div>
                {tool.description && <p className="m-0 text-xs text-muted-foreground pl-[21px]">{tool.description}</p>}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-5 gap-4 flex-wrap">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h2 className="m-0 text-lg font-semibold text-foreground">MCP Servers</h2>
            <span className="text-[10px] font-semibold text-primary bg-primary/10 rounded px-1.5 py-0.5 uppercase tracking-wider">
              Beta
            </span>
          </div>
          <div className="flex items-center gap-3">
            <p className="m-0 text-[13px] text-muted-foreground">Browse tools, authenticate once, use in chat</p>
            {loadingCounts ? (
              <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" />
                Loading tools...
              </span>
            ) : totalTools > 0 ? (
              <span className="flex items-center gap-1 text-xs text-muted-foreground">
                <Wrench className="h-3 w-3" />
                {totalTools} tool{totalTools !== 1 ? "s" : ""} available
              </span>
            ) : null}
          </div>
        </div>
        <div className="relative w-[220px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            placeholder="Search servers..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="pl-9 text-[13px] h-9"
          />
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as TabKey)} className="mb-4">
        <TabsList variant="line" className="border-b rounded-none w-full justify-start h-auto p-0">
          <TabsTrigger value="all" className="rounded-none px-4 py-2 text-[13px]">
            All
          </TabsTrigger>
          <TabsTrigger value="connected" className="rounded-none px-4 py-2 text-[13px]">
            Connected{connectedCount > 0 ? ` (${connectedCount})` : ""}
          </TabsTrigger>
        </TabsList>
      </Tabs>

      {loading ? (
        <div className="grid grid-cols-2 border rounded-lg overflow-hidden">
          {Array.from({ length: 6 }, (_, idx) => (
            <div
              key={idx}
              className={`flex items-center gap-3 p-4 ${idx % 2 === 0 ? "border-r" : ""} ${idx < 4 ? "border-b" : ""}`}
            >
              <Skeleton className="w-[38px] h-[38px] rounded-xl shrink-0" />
              <div className="flex-1 min-w-0 flex flex-col gap-1.5">
                <Skeleton className="h-3.5 w-2/3" />
                <Skeleton className="h-3 w-1/2" />
              </div>
            </div>
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="text-center text-muted-foreground text-[13px] py-12 px-3">
          {servers.length === 0
            ? "No MCP servers configured. Add servers in Tools -> MCP Servers."
            : activeTab === "connected"
              ? "No servers connected yet."
              : "No servers match your search."}
        </div>
      ) : (
        <div className="grid grid-cols-2 border rounded-lg overflow-hidden">
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
                className={`flex items-center gap-3 p-4 bg-card cursor-pointer transition-colors hover:bg-accent/30 min-w-0 ${
                  isLeftCol ? "border-r" : ""
                } ${Math.floor(idx / 2) < Math.floor((filtered.length - 1) / 2) ? "border-b" : ""}`}
              >
                {server.mcp_info?.logo_url ? (
                  <img
                    src={server.mcp_info.logo_url}
                    alt={`${name} logo`}
                    className="w-[38px] h-[38px] rounded-xl object-contain shrink-0 bg-muted/50"
                    onError={(e) => {
                      const el = e.target as HTMLImageElement;
                      el.style.display = "none";
                      if (el.nextElementSibling) (el.nextElementSibling as HTMLElement).style.display = "flex";
                    }}
                  />
                ) : null}
                <div
                  className="w-[38px] h-[38px] rounded-xl flex items-center justify-center text-white font-bold text-base shrink-0"
                  style={{
                    background: color,
                    display: server.mcp_info?.logo_url ? "none" : "flex",
                  }}
                >
                  {name.charAt(0).toUpperCase()}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-foreground truncate">{name}</div>
                  <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-1.5">
                    <span className="truncate">{server.description ?? "MCP server"}</span>
                    {count !== undefined ? (
                      count > 0 ? (
                        <span className="shrink-0 flex items-center gap-1 text-muted-foreground">
                          · <Wrench className="h-2.5 w-2.5" /> {count}
                        </span>
                      ) : null
                    ) : loadingCounts ? (
                      <Skeleton className="w-7 h-3 shrink-0" />
                    ) : null}
                  </div>
                </div>
                {(() => {
                  if (isAutoConnectedAuthType(server.auth_type)) {
                    return <CheckCircle className="h-3.5 w-3.5 text-emerald-600 shrink-0" />;
                  }
                  if (server.auth_type === AUTH_TYPE.OAUTH2) {
                    if (oauthConnected.has(server.server_id)) {
                      return <CheckCircle className="h-3.5 w-3.5 text-emerald-600 shrink-0" />;
                    }
                    return (
                      <OAuth2ConnectButton
                        server={server}
                        accessToken={accessToken}
                        onConnect={(id) => {
                          setOauthConnected((prev) => new Set(prev).add(id));
                        }}
                        variant="badge"
                      />
                    );
                  }
                  if (isConnected) {
                    return (
                      <span className="w-[7px] h-[7px] rounded-full bg-emerald-600 dark:bg-emerald-400 shrink-0" />
                    );
                  }
                  return null;
                })()}
                <ChevronRight className="h-3 w-3 text-muted-foreground/40 shrink-0" />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default MCPAppsPanel;
