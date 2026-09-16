"use client";

import { parseAsArrayOf, parseAsString, useQueryStates } from "nuqs";
import { useCallback, useMemo } from "react";

import { useUrlTableState, type UrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";
import { useUrlTab } from "@/hooks/useUrlTab";

export const PUBLIC_HUB_TABS = ["models", "agents", "mcp", "skills"] as const;
export type PublicHubTab = (typeof PUBLIC_HUB_TABS)[number];

const CLIENT_TABLE_PAGE_SIZE = 25;

const AGENT_TABLE_STATE: UrlTableStateOptions<never> = {
  sortFields: ["name", "version"],
  defaultSort: { id: "name", desc: false },
  defaultPageSize: CLIENT_TABLE_PAGE_SIZE,
  filterColumns: [],
  keyPrefix: "agent_",
  urlKeys: { search: "q" },
};

const MCP_TABLE_STATE: UrlTableStateOptions<never> = {
  sortFields: ["server_name", "transport", "auth_type"],
  defaultSort: { id: "server_name", desc: false },
  defaultPageSize: CLIENT_TABLE_PAGE_SIZE,
  filterColumns: [],
  keyPrefix: "mcp_",
  urlKeys: { search: "q" },
};

const DETAIL_PARSERS = { model: parseAsString, agent: parseAsString, mcp: parseAsString };
const DETAIL_OPTIONS = { history: "push" } as const;

const stringList = parseAsArrayOf(parseAsString).withDefault([]);
const LIST_FILTER_PARSERS = { agentSkills: stringList, mcpTransports: stringList };
const LIST_FILTER_OPTIONS = { urlKeys: { agentSkills: "agent_skills", mcpTransports: "mcp_transport" } };

export interface PublicHubAvailability {
  agentsLoading: boolean;
  hasAgents: boolean;
  mcpLoading: boolean;
  hasMcpServers: boolean;
}

export interface PublicHubUrlState {
  tab: PublicHubTab;
  setTab: (tab: PublicHubTab) => void;
  modelId: string | null;
  agentId: string | null;
  mcpId: string | null;
  openModel: (modelGroup: string) => void;
  openAgent: (agentName: string) => void;
  openMcp: (serverId: string) => void;
  closeDetail: () => void;
  agentTable: UrlTableState;
  agentSkills: string[];
  setAgentSkills: (skills: string[]) => void;
  mcpTable: UrlTableState;
  mcpTransports: string[];
  setMcpTransports: (transports: string[]) => void;
}

const toFirstPage = (table: UrlTableState) => table.onPaginationChange((page) => ({ ...page, pageIndex: 0 }));

export const usePublicHubUrlState = (availability: PublicHubAvailability): PublicHubUrlState => {
  const agents = availability.agentsLoading || availability.hasAgents;
  const mcp = availability.mcpLoading || availability.hasMcpServers;
  const visibleTabs = useMemo(() => {
    const shown: Record<PublicHubTab, boolean> = { models: true, agents, mcp, skills: true };
    return PUBLIC_HUB_TABS.filter((tab) => shown[tab]);
  }, [agents, mcp]);
  const [tab, setTab] = useUrlTab(visibleTabs, "models");
  const [details, setDetails] = useQueryStates(DETAIL_PARSERS, DETAIL_OPTIONS);
  const [filters, setFilters] = useQueryStates(LIST_FILTER_PARSERS, LIST_FILTER_OPTIONS);
  const agentTable = useUrlTableState(AGENT_TABLE_STATE);
  const mcpTable = useUrlTableState(MCP_TABLE_STATE);

  const openModel = useCallback((model: string) => void setDetails({ model }), [setDetails]);
  const openAgent = useCallback((agent: string) => void setDetails({ agent }), [setDetails]);
  const openMcp = useCallback((mcp: string) => void setDetails({ mcp }), [setDetails]);
  const closeDetail = useCallback(() => void setDetails(null), [setDetails]);

  const setAgentSkills = (agentSkills: string[]) => {
    void setFilters({ agentSkills });
    toFirstPage(agentTable);
  };
  const setMcpTransports = (mcpTransports: string[]) => {
    void setFilters({ mcpTransports });
    toFirstPage(mcpTable);
  };

  return {
    tab,
    setTab,
    modelId: details.model,
    agentId: details.agent,
    mcpId: details.mcp,
    openModel,
    openAgent,
    openMcp,
    closeDetail,
    agentTable,
    agentSkills: filters.agentSkills,
    setAgentSkills,
    mcpTable,
    mcpTransports: filters.mcpTransports,
    setMcpTransports,
  };
};
