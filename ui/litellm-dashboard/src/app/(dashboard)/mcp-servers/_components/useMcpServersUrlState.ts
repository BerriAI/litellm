import { parseAsString, parseAsStringLiteral, useQueryState, useQueryStates } from "nuqs";
import { useCallback, useEffect, useMemo, useState } from "react";
import { z } from "zod/v4";
import type { MCPServer } from "@/components/mcp_tools/types";
import { clearStorage, TOOLS_OAUTH_UI_STATE_KEY } from "@/hooks/mcpOAuthUtils";
import { useUrlTab } from "@/hooks/useUrlTab";
import { getSecureItem } from "@/utils/secureStorage";
import { EDIT_OAUTH_UI_STATE_KEY } from "./mcp_server_edit";

export const MCP_SERVER_SORT_KEYS = ["created_desc", "updated_desc", "name_asc", "health"] as const;
export type McpServerSortKey = (typeof MCP_SERVER_SORT_KEYS)[number];

const USER_PAGE_TABS = ["servers", "toolsets", "connect"] as const;
const ADMIN_PAGE_TABS = [...USER_PAGE_TABS, "semantic-filter", "tool-search", "network", "submitted"] as const;
export type McpPageTab = (typeof ADMIN_PAGE_TABS)[number];

const ADMIN_SERVER_TABS = ["overview", "tools", "settings"] as const;
export type McpServerTab = (typeof ADMIN_SERVER_TABS)[number];
const USER_SERVER_TABS: readonly McpServerTab[] = ["overview", "tools"];

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

export function useMcpServersUrlState(isAdmin: boolean, servers: readonly MCPServer[], serversLoaded: boolean) {
  const [tab, setTab] = useUrlTab<McpPageTab>(isAdmin ? ADMIN_PAGE_TABS : USER_PAGE_TABS, "servers");
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

export const useMcpServerTab = (isProxyAdmin: boolean) =>
  useUrlTab(isProxyAdmin ? ADMIN_SERVER_TABS : USER_SERVER_TABS, "overview", "server_tab");

export const useMcpToolUrlState = () => useQueryStates(toolParsers);
