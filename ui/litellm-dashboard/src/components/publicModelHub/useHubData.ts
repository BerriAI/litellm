"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import type { Plugin } from "@/components/claude_code_plugins/types";
import {
  agentHubPublicModelsCall,
  getPublicModelHubInfo,
  getUiConfig,
  mcpHubPublicServersCall,
  skillHubPublicCall,
} from "@/components/networking";
import type { AgentCard, MCPServerData } from "@/components/PublicModelHubTableColumns";

import { usePublicModelHubFacets } from "./usePublicModelHubFacets";
import { usePublicModelHubList } from "./usePublicModelHubList";

export interface HubInfo {
  docs_title: string;
  custom_docs_description: string | null;
  litellm_version: string;
  useful_links: Record<string, string | { url: string; index?: number }>;
}

const asArray = <T>(data: unknown): T[] => (Array.isArray(data) ? (data as T[]) : []);

export function useHubData() {
  const config = useQuery({ queryKey: ["publicAiHub", "uiConfig"], queryFn: () => getUiConfig(), staleTime: Infinity });
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
  const models = { ...list, isLoading: !ready || list.isLoading };
  const facets = usePublicModelHubFacets(ready);
  const [modelTotal, setModelTotal] = useState<number | undefined>(undefined);
  const settledTotal = !models.isLoading && !models.hasActiveQuery && !models.error ? models.rowCount : modelTotal;
  if (settledTotal !== modelTotal) setModelTotal(settledTotal);

  return { info, agents, mcp, skills, models, facets, modelTotal: settledTotal };
}

export type HubData = ReturnType<typeof useHubData>;
