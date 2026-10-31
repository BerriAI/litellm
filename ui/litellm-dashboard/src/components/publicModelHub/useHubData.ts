"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import type { ResourceListPage } from "@/app/(dashboard)/hooks/common/useResourceList";
import type { Plugin } from "@/components/claude_code_plugins/types";
import {
  agentHubPublicModelsCall,
  apiClient,
  getPublicModelHubInfo,
  getUiConfig,
  mcpHubPublicServersCall,
  skillHubPublicCall,
} from "@/components/networking";
import type { AgentCard, MCPServerData, ModelGroupInfo } from "@/components/PublicModelHubTableColumns";

import { usePublicModelHubFacets } from "./usePublicModelHubFacets";
import { PUBLIC_MODEL_HUB_PAGE_SIZE, PUBLIC_MODEL_HUB_PATH, usePublicModelHubList } from "./usePublicModelHubList";

export interface HubInfo {
  docs_title: string;
  custom_docs_description: string | null;
  litellm_version: string;
  useful_links: Record<string, string | { url: string; index?: number }>;
}

const asArray = <T>(data: unknown): T[] => (Array.isArray(data) ? (data as T[]) : []);

const UI_CONFIG_QUERY = {
  queryKey: ["publicAiHub", "uiConfig"],
  queryFn: () => getUiConfig(),
  staleTime: Infinity,
  retry: false,
};

export function useHubData() {
  const config = useQuery(UI_CONFIG_QUERY);
  const ready = config.isSuccess || config.isError;

  const info = useQuery<HubInfo>({
    queryKey: ["publicAiHub", "info"],
    queryFn: () => getPublicModelHubInfo(),
    enabled: ready,
  });
  const agents = useQuery<AgentCard[]>({
    queryKey: ["publicAiHub", "agents"],
    queryFn: async () => asArray<AgentCard>(await agentHubPublicModelsCall()),
    enabled: ready,
  });
  const mcp = useQuery<MCPServerData[]>({
    queryKey: ["publicAiHub", "mcp"],
    queryFn: async () => asArray<MCPServerData>(await mcpHubPublicServersCall()),
    enabled: ready,
  });
  const skills = useQuery<Plugin[]>({
    queryKey: ["publicAiHub", "skills"],
    queryFn: async () => (await skillHubPublicCall()).plugins ?? [],
    enabled: ready,
  });

  const list = usePublicModelHubList(ready);
  const listLoading = !ready || list.isLoading;
  const [listSettled, setListSettled] = useState(false);
  if (!listLoading && !listSettled) setListSettled(true);
  const models = { ...list, isLoading: listLoading && !listSettled };
  const facets = usePublicModelHubFacets(ready);
  const [modelTotal, setModelTotal] = useState<number | undefined>(undefined);
  const settledTotal = !listLoading && !models.hasActiveQuery && !models.error ? models.rowCount : modelTotal;
  if (settledTotal !== modelTotal) setModelTotal(settledTotal);

  return { info, agents, mcp, skills, models, facets, modelTotal: settledTotal };
}

export type HubData = ReturnType<typeof useHubData>;

export function useOpenedModel(open: string | null, rows: ModelGroupInfo[], enabled: boolean) {
  const onPage = rows.find((model) => model.model_group === open);
  const lookup = useQuery({
    queryKey: ["publicAiHub", "openedModel", open],
    queryFn: ({ signal }) =>
      apiClient.get<ResourceListPage<ModelGroupInfo>>(PUBLIC_MODEL_HUB_PATH, {
        query: { q: open ?? "", page_size: PUBLIC_MODEL_HUB_PAGE_SIZE },
        signal,
      }),
    enabled: enabled && open !== null && onPage === undefined,
  });
  const found = lookup.data?.data.find((model) => model.model_group === open);
  return { model: onPage ?? found, pending: onPage === undefined && lookup.isLoading };
}
