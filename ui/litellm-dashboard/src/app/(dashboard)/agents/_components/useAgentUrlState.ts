import { parseAsBoolean, parseAsString, useQueryState, useQueryStates } from "nuqs";
import { useCallback } from "react";

import { useUrlTab } from "@/hooks/useUrlTab";

const ADMIN_AGENT_TABS = ["overview", "settings"] as const;
type AgentTab = (typeof ADMIN_AGENT_TABS)[number];
const VIEWER_AGENT_TABS: readonly AgentTab[] = ["overview"];

const selectedKeyParser = parseAsString.withOptions({ history: "push" });

const AGENT_DETAIL_PARSERS = {
  agent: parseAsString.withOptions({ history: "push" }),
  tab: parseAsString,
  key: selectedKeyParser,
};

export function useAgentDetailUrlState() {
  const [{ agent: selectedAgentId }, setDetailUrl] = useQueryStates(AGENT_DETAIL_PARSERS);
  const openAgent = useCallback(
    (agentId: string) => void setDetailUrl({ agent: agentId, tab: null, key: null }),
    [setDetailUrl],
  );
  const closeAgent = useCallback(() => void setDetailUrl(null), [setDetailUrl]);
  return { selectedAgentId, openAgent, closeAgent };
}

export const useAgentHealthCheck = () => useQueryState("health_check", parseAsBoolean.withDefault(false));

export const useAgentTab = (isAdmin: boolean) => useUrlTab(isAdmin ? ADMIN_AGENT_TABS : VIEWER_AGENT_TABS, "overview");

export const useSelectedAgentKey = () => useQueryState("key", selectedKeyParser);
