import { parseAsString, useQueryStates } from "nuqs";
import { useCallback } from "react";

import { useUrlTab } from "@/hooks/useUrlTab";

export const NEW_AGENT_ID = "__new__";

const AGENT_TABS = ["configure", "chat", "batch", "connect"] as const;

export type AgentTab = (typeof AGENT_TABS)[number];

const NEW_AGENT_TABS: readonly AgentTab[] = ["configure"];

type SelectionHistory = "push" | "replace";

const agentUrlParsers = {
  agent_id: parseAsString.withOptions({ history: "push" }),
};

const resolveSelectedId = (urlId: string | null, agentIds: readonly string[]): string | null => {
  if (urlId === NEW_AGENT_ID || (urlId !== null && agentIds.includes(urlId))) return urlId;
  return agentIds[0] ?? null;
};

export function useAgentBuilderUrlState(agentIds: readonly string[]) {
  const [{ agent_id: urlId }, setAgentUrl] = useQueryStates(agentUrlParsers);
  const isNewAgent = urlId === NEW_AGENT_ID;
  const selectedId = resolveSelectedId(urlId, agentIds);
  const [activeTab, setActiveTab] = useUrlTab<AgentTab>(
    isNewAgent ? NEW_AGENT_TABS : AGENT_TABS,
    "configure",
    "agent_tab",
  );

  const selectAgent = useCallback(
    (id: string | null, history: SelectionHistory = "push") =>
      void setAgentUrl(({ agent_id: current }) => (current === id ? {} : { agent_id: id }), { history }),
    [setAgentUrl],
  );

  const dropStaleSelection = useCallback(
    (loadedIds: readonly string[]) =>
      void setAgentUrl(
        ({ agent_id: current }) =>
          current === null || current === NEW_AGENT_ID || loadedIds.includes(current) ? {} : { agent_id: null },
        { history: "replace" },
      ),
    [setAgentUrl],
  );

  return { selectedId, isNewAgent, selectAgent, dropStaleSelection, activeTab, setActiveTab };
}
