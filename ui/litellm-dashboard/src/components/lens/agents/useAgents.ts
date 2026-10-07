"use client";

import { useQuery } from "@tanstack/react-query";

import { useTracesApi } from "../traces/api";
import type { AgentSummary } from "./agentRollup";

export const AGENT_WINDOW_DAYS = 14;
const DAY_MS = 86_400_000;

/** Agents seen in the last two weeks, matching the default Braintrust project window. */
export function useAgents(accessToken: string): {
  agents: AgentSummary[];
  isLoading: boolean;
  error: Error | null;
} {
  const traces = useTracesApi(accessToken);
  const { data, isLoading, error } = useQuery({
    queryKey: ["lensAgents", accessToken, traces.live],
    queryFn: () => {
      const endMs = Date.now();
      return traces.agents({ startMs: endMs - AGENT_WINDOW_DAYS * DAY_MS, endMs });
    },
    staleTime: 60_000,
  });
  return { agents: data ?? [], isLoading, error };
}
