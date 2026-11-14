"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Search, ArrowLeft, ChevronRight, Wrench, CheckCircle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  deleteMCPOAuthUserCredential,
  fetchMCPServers,
  getMCPOAuthUserCredentialStatus,
  listMCPTools,
} from "../networking";
import {
  getMcpOAuthMode,
  MCPServer,
  MCPTool,
  handleTransport,
  isUnsupportedOnGatewayConnect,
} from "../mcp_tools/types";
import { Logo } from "@/components/molecules/logo/Logo";
import i18n from "@/i18n/i18n";
import { toast } from "@/lib/toast";
import { pendingUserMcpOAuthServerId, useUserMcpOAuthFlow, type UserMcpOAuthStatus } from "@/hooks/useUserMcpOAuthFlow";
import { getSecureItem, setSecureItem } from "@/utils/secureStorage";

interface OAuth2ConnectButtonProps {
  server: Pick<MCPServer, "server_id" | "server_name" | "alias">;
  accessToken: string;
  onConnect: (serverId: string) => void;
  variant?: "badge" | "button";
  autoStartKey?: string | null;
  scopes?: string[];
  buttonLabel?: string;
}

export const OAuth2ConnectButton: React.FC<OAuth2ConnectButtonProps> = ({
  server,
  accessToken,
  onConnect,
  variant = "badge",
  autoStartKey = null,
  scopes,
  buttonLabel = i18n.t("chat.mcpAppsPanel.connect"),
}) => {
  const { t } = useTranslation();
  const name = server.server_name ?? server.alias ?? server.server_id;
  const flowOptions = {
    accessToken,
    serverId: server.server_id,
    serverAlias: name,
    scopes,
    onSuccess: useCallback(() => onConnect(server.server_id), [onConnect, server.server_id]),
  };
  const { startOAuthFlow, status } = useUserMcpOAuthFlow(flowOptions);

  useEffect(() => {
    if (autoStartKey === null || status !== "idle" || getSecureItem(autoStartKey) !== null) return;
    setSecureItem(autoStartKey, "1");
    startOAuthFlow();
  }, [autoStartKey, status, startOAuthFlow]);

  const loading = status === "authorizing" || status === "exchanging";

  if (variant === "button") {
    return (
      <Button onClick={startOAuthFlow} disabled={loading} className="font-semibold h-[38px] min-w-[110px]">
        {loading && <Loader2 className="h-4 w-4 animate-spin mr-1.5" />}
        {loading ? t("chat.mcpAppsPanel.connecting") : buttonLabel}
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
      {loading ? t("chat.mcpAppsPanel.connecting") : t("chat.mcpAppsPanel.connect")}
    </span>
  );
};

export interface ConnectedServers {
  on: string[];
  all: string[];
}

export type ServerListReport = ConnectedServers | "loading" | "failed" | "saving-login";

const savingReturnedLogin = (serverId: string, status: UserMcpOAuthStatus): boolean =>
  serverId !== "" && (status === "idle" || status === "exchanging");

const listReport = (
  loading: boolean,
  loadFailed: boolean,
  savingLogin: boolean,
  servers: ConnectedServers,
): ServerListReport => {
  if (savingLogin) return "saving-login";
  if (loading) return "loading";
  if (loadFailed) return "failed";
  return servers;
};

interface Props {
  accessToken: string;
  selectedServers: string[];
  onChange: (servers: string[]) => void;
  connectMode?: boolean;
  offServerIds?: readonly string[];
  onOffServerIdsChange?: (serverIds: string[]) => void;
  lockedOn?: LockedOn;
  onConnectedServers?: (servers: ServerListReport) => void;
}

const NO_SERVER_IDS: readonly string[] = [];

export type LockedOn = readonly string[] | "all" | undefined;

const isLockedOn = (lockedOn: LockedOn, serverId: string): boolean =>
  lockedOn === "all" || (lockedOn?.includes(serverId) ?? false);

const CHAT_LABELS = {
  tab: "chat.mcpAppsPanel.tabs.connected",
  on: "chat.mcpAppsPanel.info.connected",
  off: "chat.mcpAppsPanel.info.notConnected",
} as const;
const CONNECT_LABELS = {
  tab: "chat.mcpAppsPanel.tabs.on",
  on: "chat.mcpAppsPanel.info.on",
  off: "chat.mcpAppsPanel.info.off",
} as const;
const labelsFor = (connectMode: boolean | undefined) => (connectMode ? CONNECT_LABELS : CHAT_LABELS);

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

const MCPAppsPanel: React.FC<Props> = ({
  accessToken,
  selectedServers,
  onChange,
  connectMode,
  offServerIds = NO_SERVER_IDS,
  onOffServerIdsChange,
  lockedOn,
  onConnectedServers,
}) => {
  const { t } = useTranslation();
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadFailed, setLoadFailed] = useState(false);
  const [query, setQuery] = useState("");
  const [activeTab, setActiveTab] = useState<TabKey>("all");
  const [togglingOn, setTogglingOn] = useState<Set<string>>(new Set());
  const [detailServerId, setDetailServerId] = useState<string | null>(null);
  const [toolCounts, setToolCounts] = useState<Record<string, number>>({});
  const [loadingCounts, setLoadingCounts] = useState(false);
  const [oauthConnected, setOauthConnected] = useState<Set<string>>(new Set());
  const [oauthChecking, setOauthChecking] = useState<Set<string>>(new Set());

  const serversRef = useRef<MCPServer[]>([]);
  const commitServers = useCallback((next: MCPServer[]) => {
    serversRef.current = next;
    setServers(next);
  }, []);
  const selectedServersRef = useRef<string[]>(selectedServers);
  useEffect(() => {
    selectedServersRef.current = selectedServers;
  }, [selectedServers]);
  const onChangeRef = useRef(onChange);
  useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  const nameOf = (s: MCPServer) => s.server_name ?? s.alias ?? s.server_id;

  const detailServer = servers.find((s) => s.server_id === detailServerId);

  const connectUnavailabilityLabel = useCallback(
    (s: MCPServer): string | null => {
      if (!connectMode) return null;
      if (isUnsupportedOnGatewayConnect(s.auth_type)) return t("chat.mcpAppsPanel.notSupported");
      return null;
    },
    [connectMode, t],
  );

  const connectableNow = useCallback(
    (serverId: string): MCPServer | undefined => {
      const current = serversRef.current.find((s) => s.server_id === serverId);
      return current !== undefined && connectUnavailabilityLabel(current) === null ? current : undefined;
    },
    [connectUnavailabilityLabel],
  );

  const [returningServerId] = useState(() => (connectMode ? pendingUserMcpOAuthServerId() : null) ?? "");
  const returningOAuth = useUserMcpOAuthFlow({
    accessToken,
    serverId: returningServerId,
    onSuccess: useCallback(() => {
      setOauthConnected((prev) => new Set(prev).add(returningServerId));
    }, [returningServerId]),
  });

  const fetchToolCount = useCallback(
    async (server: MCPServer, isCurrentLoad: () => boolean) => {
      try {
        const toolsData = await listMCPTools(accessToken, server.server_id);
        if (!isCurrentLoad()) return;
        const tools: MCPTool[] = Array.isArray(toolsData?.tools) ? toolsData.tools : [];
        setToolCounts((prev) => ({ ...prev, [nameOf(server)]: tools.length }));
      } catch {
        // ignore
      }
    },
    [accessToken],
  );

  const checkOauthCredential = useCallback(
    async (server: MCPServer, isCurrentLoad: () => boolean) => {
      try {
        const status = await getMCPOAuthUserCredentialStatus(accessToken, server.server_id);
        if (!isCurrentLoad()) return;
        if (status.has_credential && !status.is_expired) {
          setOauthConnected((prev) => new Set(prev).add(server.server_id));
        }
      } catch {
        // ignore
      } finally {
        if (isCurrentLoad()) {
          setOauthChecking((prev) => {
            const next = new Set(prev);
            next.delete(server.server_id);
            return next;
          });
        }
      }
    },
    [accessToken],
  );

  useEffect(() => {
    let current = true;
    const isCurrentLoad = () => current;

    fetchMCPServers(accessToken, undefined, connectMode)
      .then(async (serverData) => {
        if (!isCurrentLoad()) return;
        const list: MCPServer[] = Array.isArray(serverData) ? serverData : serverData?.data ?? [];
        const reachable = connectMode ? list.filter((s) => s.connected_app_reachable !== false) : list;
        const oauthServers = reachable.filter((s) => getMcpOAuthMode(s) === "authorization_code");
        commitServers(reachable);
        setOauthChecking(new Set(oauthServers.map((s) => s.server_id)));
        setLoadFailed(false);
        setLoading(false);

        oauthServers.forEach((s) => checkOauthCredential(s, isCurrentLoad));

        setLoadingCounts(true);
        const chunks = Array.from({ length: Math.ceil(reachable.length / TOOLS_FETCH_CONCURRENCY) }, (_, i) =>
          reachable.slice(i * TOOLS_FETCH_CONCURRENCY, (i + 1) * TOOLS_FETCH_CONCURRENCY),
        );
        for (const chunk of chunks) {
          if (!isCurrentLoad()) return;
          await Promise.allSettled(chunk.map((s) => fetchToolCount(s, isCurrentLoad)));
        }
        if (isCurrentLoad()) setLoadingCounts(false);
      })
      .catch(() => {
        if (isCurrentLoad()) {
          commitServers([]);
          setLoadFailed(true);
          setLoading(false);
        }
      });
    return () => {
      current = false;
    };
  }, [accessToken, connectMode, commitServers, fetchToolCount, checkOauthCredential]);

  useEffect(() => {
    if (oauthConnected.size === 0) return;
    const namesToAdd = serversRef.current
      .filter(
        (s) =>
          oauthConnected.has(s.server_id) &&
          !selectedServersRef.current.includes(nameOf(s)) &&
          connectUnavailabilityLabel(s) === null,
      )
      .map(nameOf);
    if (namesToAdd.length > 0) {
      onChangeRef.current([...selectedServersRef.current, ...namesToAdd]);
    }
  }, [oauthConnected, connectUnavailabilityLabel]);

  const handleToggle = async (server: MCPServer, checked: boolean) => {
    const serverName = nameOf(server);
    if (!checked) {
      onChange(selectedServers.filter((s) => s !== serverName));
      setOauthConnected((prev) => {
        const next = new Set(prev);
        next.delete(server.server_id);
        return next;
      });
      return;
    }
    if (connectableNow(server.server_id) === undefined) return;
    setTogglingOn((prev) => new Set(prev).add(serverName));
    try {
      const result = await listMCPTools(accessToken, server.server_id);
      if (result?.error) {
        toast.warning(t("chat.mcpAppsPanel.toolsLoadFailed", { serverName }));
        return;
      }
      if (connectableNow(server.server_id) === undefined) return;
      if (!selectedServersRef.current.includes(serverName)) {
        onChange([...selectedServersRef.current, serverName]);
      }
    } catch {
      toast.warning(t("chat.mcpAppsPanel.toolsLoadFailed", { serverName }));
    } finally {
      setTogglingOn((prev) => {
        const next = new Set(prev);
        next.delete(serverName);
        return next;
      });
    }
  };

  const isOn = (server: MCPServer) => {
    if (connectUnavailabilityLabel(server) !== null) return false;
    if (!connectMode) return selectedServers.includes(nameOf(server));
    return isLockedOn(lockedOn, server.server_id) || !offServerIds.includes(server.server_id);
  };

  const setServerOn = (server: MCPServer, on: boolean) =>
    onOffServerIdsChange?.(
      on ? offServerIds.filter((id) => id !== server.server_id) : [...offServerIds, server.server_id],
    );

  const renderServerSwitch = (server: MCPServer) => (
    <span className="ml-2 inline-flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
      {lockedOn !== "all" && isLockedOn(lockedOn, server.server_id) && (
        <span className="text-[11px] text-muted-foreground whitespace-nowrap">
          {t("chat.mcpAppsPanel.requiredByClient")}
        </span>
      )}
      <Switch
        checked={isOn(server)}
        disabled={isLockedOn(lockedOn, server.server_id)}
        onCheckedChange={(checked) => setServerOn(server, checked)}
        aria-label={t("chat.mcpAppsPanel.useServer", { serverName: nameOf(server) })}
      />
    </span>
  );

  const renderLoginState = (server: MCPServer) => {
    const mode = getMcpOAuthMode(server);
    if (mode === "m2m") return <CheckCircle className="h-3.5 w-3.5 text-success shrink-0" />;
    if (mode !== "authorization_code") return null;
    if (oauthConnected.has(server.server_id)) {
      return <CheckCircle className="h-3.5 w-3.5 text-success shrink-0" />;
    }
    if (
      oauthChecking.has(server.server_id) ||
      (returningOAuth.status === "exchanging" && server.server_id === returningServerId)
    ) {
      return <Skeleton className="h-6 w-16 shrink-0 rounded-md" />;
    }
    return (
      <>
        {connectMode && (
          <span className="text-[11px] text-muted-foreground shrink-0 whitespace-nowrap">
            {t("chat.mcpAppsPanel.signInToUse")}
          </span>
        )}
        <OAuth2ConnectButton
          server={server}
          accessToken={accessToken}
          onConnect={(id) => setOauthConnected((prev) => new Set(prev).add(id))}
          variant="badge"
        />
      </>
    );
  };

  const renderConnectionIndicator = (server: MCPServer) => {
    const unavailabilityLabel = connectUnavailabilityLabel(server);
    if (unavailabilityLabel !== null) {
      return (
        <span className="text-[11px] text-muted-foreground shrink-0 whitespace-nowrap">{unavailabilityLabel}</span>
      );
    }
    const loginState = renderLoginState(server);
    if (connectMode) {
      return (
        <div className="flex items-center gap-2 shrink-0">
          {loginState}
          {renderServerSwitch(server)}
        </div>
      );
    }
    if (loginState !== null) return loginState;
    if (selectedServers.includes(nameOf(server))) {
      return <span className="w-[7px] h-[7px] rounded-full bg-success shrink-0" />;
    }
    return null;
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
    const matchesTab = activeTab === "all" || isOn(s);
    return matchesQuery && matchesTab;
  });

  const connectedCount = servers.filter(isOn).length;
  const labels = labelsFor(connectMode);

  const connectedServers = useMemo(() => {
    const all = servers.filter((s) => connectUnavailabilityLabel(s) === null).map((s) => s.server_id);
    return { on: all.filter((id) => isLockedOn(lockedOn, id) || !offServerIds.includes(id)), all };
  }, [servers, offServerIds, lockedOn, connectUnavailabilityLabel]);
  useEffect(() => {
    onConnectedServers?.(
      listReport(loading, loadFailed, savingReturnedLogin(returningServerId, returningOAuth.status), connectedServers),
    );
  }, [loading, loadFailed, returningServerId, returningOAuth.status, connectedServers, onConnectedServers]);

  const emptyStateText = () => {
    if (loadFailed && connectMode) return t("chat.mcpAppsPanel.empty.loadFailed");
    if (servers.length === 0) {
      return connectMode ? t("chat.mcpAppsPanel.empty.connectMode") : t("chat.mcpAppsPanel.empty.noServers");
    }
    return activeTab === "connected"
      ? t("chat.mcpAppsPanel.empty.noneConnected")
      : t("chat.mcpAppsPanel.empty.noMatch");
  };
  const totalTools = Object.values(toolCounts).reduce((sum, n) => sum + n, 0);

  if (detailServer) {
    const name = nameOf(detailServer);
    const isConnected = isOn(detailServer);
    const isTogglingOn = togglingOn.has(name);
    const color = getAvatarColor(name);

    const renderDetailLogin = () => {
      if (getMcpOAuthMode(detailServer) === "m2m") {
        return <span className="text-[13px] text-muted-foreground">{t("chat.mcpAppsPanel.authorized")}</span>;
      }
      if (getMcpOAuthMode(detailServer) !== "authorization_code") {
        if (connectMode) return null;
        return (
          <Button
            variant={isConnected ? "outline" : "default"}
            disabled={isTogglingOn}
            onClick={() => handleToggle(detailServer, !isConnected)}
            className="font-semibold h-[38px] min-w-[110px]"
          >
            {isTogglingOn && <Loader2 className="h-4 w-4 animate-spin mr-1.5" />}
            {isConnected ? t("chat.mcpAppsPanel.disconnect") : t("chat.mcpAppsPanel.connect")}
          </Button>
        );
      }
      if (oauthConnected.has(detailServer.server_id)) {
        return (
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
              if (!connectMode) onChangeRef.current(selectedServersRef.current.filter((s) => s !== name));
            }}
            className="font-semibold h-[38px] min-w-[110px]"
          >
            {t("chat.mcpAppsPanel.disconnect")}
          </Button>
        );
      }
      return (
        <OAuth2ConnectButton
          server={detailServer}
          accessToken={accessToken}
          onConnect={(id) => {
            setOauthConnected((prev) => new Set(prev).add(id));
          }}
          variant="button"
        />
      );
    };

    const renderDetailAction = () => {
      const unavailabilityLabel = connectUnavailabilityLabel(detailServer);
      if (unavailabilityLabel !== null) {
        return <span className="text-[13px] text-muted-foreground py-2.5 shrink-0">{unavailabilityLabel}</span>;
      }
      if (!connectMode) return renderDetailLogin();
      return (
        <div className="flex items-center gap-3 shrink-0">
          {renderDetailLogin()}
          {renderServerSwitch(detailServer)}
        </div>
      );
    };

    return (
      <div className="w-full">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setDetailServerId(null)}
          className="-ml-3 mb-5 gap-1.5 text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3 w-3" />
          {t("chat.mcpAppsPanel.back")}
        </Button>

        <div className="flex items-start gap-5 mb-7">
          {detailServer.mcp_info?.logo_url ? (
            <Logo
              src={detailServer.mcp_info.logo_url}
              label={name}
              className="w-16 h-16 rounded-2xl object-contain shrink-0 bg-muted/50"
            />
          ) : (
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center text-white font-bold text-[28px] shrink-0"
              style={{ background: color }}
            >
              {name.charAt(0).toUpperCase()}
            </div>
          )}
          <div className="flex-1">
            <h2 className="m-0 mb-1 text-[22px] font-bold text-foreground">{name}</h2>
            <p className="m-0 text-sm text-muted-foreground">
              {detailServer.description ?? t("chat.mcpAppsPanel.defaultDescription")}
            </p>
          </div>
          {renderDetailAction()}
        </div>

        <h3 className="m-0 mb-3 text-[15px] font-semibold text-foreground">{t("chat.mcpAppsPanel.information")}</h3>
        <div className="border rounded-lg overflow-hidden mb-7">
          {[
            [t("chat.mcpAppsPanel.info.serverId"), detailServer.server_id],
            [t("chat.mcpAppsPanel.info.transport"), handleTransport(detailServer.transport, detailServer.spec_path)],
            [t("chat.mcpAppsPanel.info.status"), isConnected ? t(labels.on) : t(labels.off)],
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
          <h3 className="m-0 text-[15px] font-semibold text-foreground">{t("chat.mcpAppsPanel.availableTools")}</h3>
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
          <div className="text-muted-foreground text-[13px] py-2">{t("chat.mcpAppsPanel.noTools")}</div>
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
            <h2 className="m-0 text-lg font-semibold text-foreground">{t("chat.mcpAppsPanel.title")}</h2>
            {!connectMode && (
              <span className="text-[10px] font-semibold text-primary bg-primary/10 rounded px-1.5 py-0.5 uppercase tracking-wider">
                {t("chat.mcpAppsPanel.beta")}
              </span>
            )}
          </div>
          {connectMode ? (
            <p className="m-0 text-[13px] text-muted-foreground">{t("chat.mcpAppsPanel.connectModeSubtitle")}</p>
          ) : (
            <div className="flex items-center gap-3">
              <p className="m-0 text-[13px] text-muted-foreground">{t("chat.mcpAppsPanel.subtitle")}</p>
              {loadingCounts ? (
                <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  {t("chat.mcpAppsPanel.loadingTools")}
                </span>
              ) : totalTools > 0 ? (
                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Wrench className="h-3 w-3" />
                  {t("chat.mcpAppsPanel.toolsAvailable", { count: totalTools })}
                </span>
              ) : null}
            </div>
          )}
        </div>
        <div className="relative w-[220px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            placeholder={t("chat.mcpAppsPanel.searchPlaceholder")}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="pl-9 text-[13px] h-9"
          />
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as TabKey)} className="mb-4">
        <TabsList variant="line" className="border-b rounded-none w-full justify-start h-auto p-0">
          <TabsTrigger value="all" className="rounded-none px-4 py-2 text-[13px]">
            {t("chat.mcpAppsPanel.tabs.all")}
          </TabsTrigger>
          <TabsTrigger value="connected" className="rounded-none px-4 py-2 text-[13px]">
            {t(labels.tab)}
            {connectedCount > 0 ? ` (${connectedCount})` : ""}
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
        <div className="text-center text-muted-foreground text-[13px] py-12 px-3">{emptyStateText()}</div>
      ) : (
        <div className="grid grid-cols-2 border rounded-lg overflow-hidden">
          {filtered.map((server, idx) => {
            const name = nameOf(server);
            const color = getAvatarColor(name);
            const isLeftCol = idx % 2 === 0;
            const count = toolCounts[name];
            const unavailable = connectUnavailabilityLabel(server) !== null;

            return (
              <div
                key={server.server_id}
                onClick={() => setDetailServerId(server.server_id)}
                className={`flex items-center gap-3 p-4 bg-card cursor-pointer transition-colors hover:bg-accent/30 min-w-0 ${
                  isLeftCol ? "border-r" : ""
                } ${Math.floor(idx / 2) < Math.floor((filtered.length - 1) / 2) ? "border-b" : ""} ${
                  unavailable ? "opacity-50" : ""
                }`}
              >
                {server.mcp_info?.logo_url ? (
                  <Logo
                    src={server.mcp_info.logo_url}
                    label={name}
                    className="w-[38px] h-[38px] rounded-xl object-contain shrink-0 bg-muted/50"
                  />
                ) : (
                  <div
                    className="w-[38px] h-[38px] rounded-xl flex items-center justify-center text-white font-bold text-base shrink-0"
                    style={{ background: color }}
                  >
                    {name.charAt(0).toUpperCase()}
                  </div>
                )}
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-foreground truncate">{name}</div>
                  <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-1.5">
                    <span className="truncate">{server.description ?? t("chat.mcpAppsPanel.defaultDescription")}</span>
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
                {renderConnectionIndicator(server)}
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
