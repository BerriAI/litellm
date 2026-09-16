"use client";

import { parseAsString, useQueryStates } from "nuqs";
import { useCallback } from "react";

import { useUrlTableState, type UrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";
import {
  type ModelFiltersState,
  type ModelFilterValues,
  useModelFiltersUrlState,
} from "@/components/useModelFiltersState";
import { useUrlTab } from "@/hooks/useUrlTab";

export const AI_HUB_TABS = ["models", "agents", "mcp", "skills"] as const;
export type AiHubTab = (typeof AI_HUB_TABS)[number];

const PAGE_SIZE = 25;

const MODELS_TABLE_STATE: UrlTableStateOptions<never> = {
  sortFields: ["model_group", "providers", "mode", "max_input_tokens", "input_cost_per_token", "is_public_model_group"],
  defaultSort: { id: "model_group", desc: false },
  defaultPageSize: PAGE_SIZE,
  filterColumns: [],
  keyPrefix: "models_",
};

const AGENTS_TABLE_STATE: UrlTableStateOptions<never> = {
  sortFields: ["name", "description", "version", "protocolVersion", "is_public"],
  defaultSort: { id: "name", desc: false },
  defaultPageSize: PAGE_SIZE,
  filterColumns: [],
  urlKeys: {
    search: "agent_q",
    sort_by: "agents_sort_by",
    sort_order: "agents_sort_order",
    page: "agents_page",
    page_size: "agents_page_size",
  },
};

const MCP_TABLE_STATE: UrlTableStateOptions<never> = {
  sortFields: ["server_name", "description", "transport", "auth_type", "status", "created_by", "is_public"],
  defaultSort: { id: "server_name", desc: false },
  defaultPageSize: PAGE_SIZE,
  filterColumns: [],
  keyPrefix: "mcp_",
};

const DETAIL_PARSERS = { model: parseAsString, agent: parseAsString, mcp: parseAsString };
const DETAIL_OPTIONS = { history: "push" } as const;

export interface ModelHubTableUrlState {
  tab: AiHubTab;
  setTab: (tab: AiHubTab) => void;
  modelId: string | null;
  agentId: string | null;
  mcpId: string | null;
  openModel: (modelGroup: string) => void;
  openAgent: (agentId: string) => void;
  openMcp: (serverId: string) => void;
  closeDetail: () => void;
  modelFilters: ModelFiltersState;
  modelsTable: UrlTableState;
  agentsTable: UrlTableState;
  mcpTable: UrlTableState;
}

export const useModelHubTableUrlState = (): ModelHubTableUrlState => {
  const [tab, setTab] = useUrlTab(AI_HUB_TABS, "models");
  const [details, setDetails] = useQueryStates(DETAIL_PARSERS, DETAIL_OPTIONS);
  const urlFilters = useModelFiltersUrlState();
  const modelsTable = useUrlTableState(MODELS_TABLE_STATE);
  const agentsTable = useUrlTableState(AGENTS_TABLE_STATE);
  const mcpTable = useUrlTableState(MCP_TABLE_STATE);

  const openModel = useCallback((model: string) => void setDetails({ model }), [setDetails]);
  const openAgent = useCallback((agent: string) => void setDetails({ agent }), [setDetails]);
  const openMcp = useCallback((mcp: string) => void setDetails({ mcp }), [setDetails]);
  const closeDetail = useCallback(() => void setDetails(null), [setDetails]);

  const toFirstModelsPage = () => modelsTable.onPaginationChange((page) => ({ ...page, pageIndex: 0 }));
  const modelFilters: ModelFiltersState = {
    values: urlFilters.values,
    update: (patch: Partial<ModelFilterValues>) => {
      urlFilters.update(patch);
      toFirstModelsPage();
    },
    reset: () => {
      urlFilters.reset();
      toFirstModelsPage();
    },
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
    modelFilters,
    modelsTable,
    agentsTable,
    mcpTable,
  };
};
