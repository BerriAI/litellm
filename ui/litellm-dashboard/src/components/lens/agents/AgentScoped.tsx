"use client";

import { useTracesLive } from "../traces/api";
import { AgentPicker } from "./AgentPicker";
import { useAgents } from "./useAgents";
import { useAgentSelection } from "./useAgentSelection";

export interface LensAgents {
  readonly agent: string | null;
  select(agent: string): void;
  readonly list: ReturnType<typeof useAgents>;
}

export function useLensAgents(accessToken: string): LensAgents {
  const list = useAgents(accessToken);
  const { agent, select } = useAgentSelection(
    !useTracesLive(),
    list.agents.map((item) => item.name),
  );
  return { agent, select, list };
}

/** `Lens / agent ▾`, shown whenever there is an agent to scope to. */
export function AgentBreadcrumb({ agents }: { agents: LensAgents }) {
  if (!agents.agent) return null;
  return (
    <>
      <span aria-hidden className="text-muted-foreground">
        /
      </span>
      <AgentPicker agent={agents.agent} agents={agents.list.agents} onSelect={agents.select} />
    </>
  );
}
