import { parseAsString, parseAsStringLiteral, useQueryState, useQueryStates } from "nuqs";
import { useCallback, useEffect, useMemo, useState } from "react";
import { z } from "zod";
import type { MCPServer } from "@/components/mcp_tools/types";
import { clearStorage, TOOLS_OAUTH_UI_STATE_KEY } from "@/hooks/mcpOAuthUtils";
import { useUrlTab } from "@/hooks/useUrlTab";
import { getSecureItem } from "@/utils/secureStorage";
import { EDIT_OAUTH_UI_STATE_KEY } from "./mcp_server_edit";

export const MCP_SERVER_SORT_KEYS = ["created_desc", "updated_desc", "name_asc", "health"] as const;
export type McpServerSortKey = (typeof MCP_SERVER_SORT_KEYS)[number];

export const PAGE_TABS = [
  "servers",
  "toolsets",
  "connect",
  "semantic-filter",
  "tool-search",
  "network-settings",
  "submitted",
  "connections",
] as const;
export type McpPageTab = (typeof PAGE_TABS)[number];

export const SERVER_TABS = ["overview", "tools", "settings", "user-credentials"] as const;
export type McpServerTab = (typeof SERVER_TABS)[number];

export interface McpPageAccess {
  isAdmin: boolean;
  canViewLiveConnections: boolean;
}

export interface McpServerAccess {
  isProxyAdmin: boolean;
  canViewUserCredentials: boolean;
}

const PAGE_TAB_GATES: Record<McpPageTab, (access: McpPageAccess) => boolean> = {
  servers: () => true,
  toolsets: () => true,
  connect: () => true,
  "semantic-filter": ({ isAdmin }) => isAdmin,
  "tool-search": ({ isAdmin }) => isAdmin,
  "network-settings": ({ isAdmin }) => isAdmin,
  submitted: ({ isAdmin }) => isAdmin,
  connections: ({ canViewLiveConnections }) => canViewLiveConnections,
};

const SERVER_TAB_GATES: Record<McpServerTab, (access: McpServerAccess) => boolean> = {
  overview: () => true,
  tools: () => true,
  settings: ({ isProxyAdmin }) => isProxyAdmin,
  "user-credentials": ({ canViewUserCredentials }) => canViewUserCredentials,
};

export const visiblePageTabs = (access: McpPageAccess): readonly McpPageTab[] =>
  PAGE_TABS.filter((tab) => PAGE_TAB_GATES[tab](access));

export const visibleServerTabs = (access: McpServerAccess): readonly McpServerTab[] =>
  SERVER_TABS.filter((tab) => SERVER_TAB_GATES[tab](access));

export const resolveVisibleTab = <T extends string>(requested: T, visible: readonly T[], fallback: T): T =>
  visible.includes(requested) ? requested : fallback;

const listParsers = {
  team: parseAsString.withDefault("all"),
  access_group: parseAsString.withDefault("all"),
  search: parseAsString.withDefault(""),
  sort: parseAsStringLiteral(MCP_SERVER_SORT_KEYS).withDefault("created_desc"),
};

const detailParsers = {
  server: parseAsString,
  server_tab: parseAsString,
  tool: parseAsString,
  tool_search: parseAsString,
};

const CLOSED_DETAIL = { server: null, server_tab: null, tool: null, tool_search: null };

const toolParsers = {
  tool: parseAsString.withOptions({ history: "push" }),
  tool_search: parseAsString.withDefault(""),
};

const storedServerSchema = z.object({ serverId: z.string().min(1) });

interface OAuthRestore {
  serverId: string;
  tab: McpServerTab;
}

const readStoredServerId = (key: string): string | null => {
  const stored = getSecureItem(key);
  if (!stored) return null;
  try {
    const parsed = storedServerSchema.safeParse(JSON.parse(stored));
    return parsed.success ? parsed.data.serverId : null;
  } catch {
    return null;
  }
};

const readOAuthRestore = (): OAuthRestore | null => {
  if (typeof window === "undefined") return null;
  const editServerId = readStoredServerId(EDIT_OAUTH_UI_STATE_KEY);
  if (editServerId) return { serverId: editServerId, tab: "settings" };
  const toolsServerId = readStoredServerId(TOOLS_OAUTH_UI_STATE_KEY);
  return toolsServerId ? { serverId: toolsServerId, tab: "tools" } : null;
};

const toSortKey = (value: string | null): McpServerSortKey | null =>
  MCP_SERVER_SORT_KEYS.find((key) => key === value) ?? null;

export function useMcpServersUrlState(access: McpPageAccess, servers: readonly MCPServer[], serversLoaded: boolean) {
  const [requestedTab, setTab] = useUrlTab(PAGE_TABS, "servers");
  const tab = resolveVisibleTab(requestedTab, visiblePageTabs(access), "servers");
  const [filters, setFilters] = useQueryStates(listParsers);
  const [detail, setDetail] = useQueryStates(detailParsers);
  const [fillEnvVars, setFillEnvVars] = useQueryState("fill_env_vars", parseAsString);
  const [envVarsDeepLinkId, setEnvVarsDeepLinkId] = useState(fillEnvVars);
  const [oauthRestore] = useState(() => {
    const stored = readOAuthRestore();
    return stored && (detail.server ?? stored.serverId) === stored.serverId ? stored : null;
  });
  const [editServer, setEditServer] = useState(oauthRestore?.tab === "settings");
  const restoredServerMissing =
    oauthRestore !== null && serversLoaded && !servers.some((server) => server.server_id === oauthRestore.serverId);

  useEffect(() => {
    clearStorage(TOOLS_OAUTH_UI_STATE_KEY);
    if (!oauthRestore) {
      clearStorage(EDIT_OAUTH_UI_STATE_KEY);
      return;
    }
    void setDetail({ server: oauthRestore.serverId, server_tab: oauthRestore.tab });
  }, [oauthRestore, setDetail]);

  useEffect(() => {
    if (restoredServerMissing) clearStorage(EDIT_OAUTH_UI_STATE_KEY);
  }, [restoredServerMissing]);

  useEffect(() => {
    if (fillEnvVars !== null) void setFillEnvVars(null);
  }, [fillEnvVars, setFillEnvVars]);

  const selectedServer = useMemo(
    () => servers.find((server) => server.server_id === detail.server) ?? null,
    [servers, detail.server],
  );

  const openServer = useCallback(
    (serverId: string) => {
      setEditServer(true);
      void setDetail({ ...CLOSED_DETAIL, server: serverId }, { history: "push" });
    },
    [setDetail],
  );

  const closeServer = useCallback(() => {
    setEditServer(false);
    void setDetail(CLOSED_DETAIL, { history: "push" });
  }, [setDetail]);

  return {
    tab,
    setTab,
    team: filters.team,
    setTeam: (team: string | null) => void setFilters({ team }),
    accessGroup: filters.access_group,
    setAccessGroup: (group: string | null) => void setFilters({ access_group: group }),
    search: filters.search,
    setSearch: (search: string) => void setFilters({ search }),
    sort: filters.sort,
    setSort: (sort: string | null) => void setFilters({ sort: toSortKey(sort) }),
    selectedServerId: detail.server,
    selectedServer,
    editServer,
    openServer,
    closeServer,
    envVarsDeepLinkId,
    clearEnvVarsDeepLink: () => setEnvVarsDeepLinkId(null),
  };
}

export const useMcpServerTab = (access: McpServerAccess): [McpServerTab, (tab: McpServerTab) => void] => {
  const [requestedTab, setServerTab] = useUrlTab(SERVER_TABS, "overview", "server_tab");
  return [resolveVisibleTab(requestedTab, visibleServerTabs(access), "overview"), setServerTab];
};

export const useMcpToolUrlState = () => useQueryStates(toolParsers);
