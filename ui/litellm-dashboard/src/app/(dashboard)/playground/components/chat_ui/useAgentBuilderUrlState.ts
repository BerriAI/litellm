import { parseAsString, useQueryStates } from "nuqs";
import { useCallback } from "react";

import { useUrlTab } from "@/hooks/useUrlTab";

export const NEW_AGENT_ID = "__new__";

const AGENT_TABS = ["configure", "chat", "batch", "connect"] as const;

export type AgentTab = (typeof AGENT_TABS)[number];

const NEW_AGENT_TABS: readonly AgentTab[] = ["configure"];

const agentUrlParsers = {
  agent_id: parseAsString.withOptions({ history: "push" }),
};

export function useAgentBuilderUrlState() {
  const [{ agent_id: selectedId }, setAgentUrl] = useQueryStates(agentUrlParsers);
  const isNewAgent = selectedId === NEW_AGENT_ID;
  const [activeTab, setActiveTab] = useUrlTab<AgentTab>(
    isNewAgent ? NEW_AGENT_TABS : AGENT_TABS,
    "configure",
    "agent_tab",
  );

  const selectAgent = useCallback((id: string | null) => void setAgentUrl({ agent_id: id }), [setAgentUrl]);

  const keepValidSelection = useCallback(
    (agentIds: readonly string[]) =>
      void setAgentUrl(
        ({ agent_id: current }) => {
          const stillValid = current === NEW_AGENT_ID || (current !== null && agentIds.includes(current));
          const fallback = agentIds[0] ?? null;
          return stillValid || current === fallback ? {} : { agent_id: fallback };
        },
        { history: "replace" },
      ),
    [setAgentUrl],
  );

  return { selectedId, isNewAgent, selectAgent, keepValidSelection, activeTab, setActiveTab };
}
