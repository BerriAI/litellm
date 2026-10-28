import { useCallback, useEffect, useState } from "react";
import { dailyActivityExportCall } from "@/components/networking";
import type { DailyActivityRequest } from "@/components/UsagePage/dailyActivityApi";
import type { AgentRow } from "../components/overview/agentCatalog";
import { clientAgentModelsFromExport, type ClientAgentModelsResult } from "./clientAgentModelsData";

interface ClientAgentModelsOptions {
  accessToken: string | null;
  startTime: Date | null;
  endTime: Date | null;
  agent: AgentRow | null;
  enabled: boolean;
}

export const MAX_CLIENT_MODEL_EXPORT_BYTES = 20 * 1024 * 1024;

const loadModels = async (request: DailyActivityRequest, agent: AgentRow): Promise<ClientAgentModelsResult> => {
  try {
    const blob = await dailyActivityExportCall("tag", request, "daily_with_models", "json");
    if (blob.size > MAX_CLIENT_MODEL_EXPORT_BYTES) {
      return { results: [], error: "Choose a shorter date range to load the model breakdown" };
    }
    const payload: unknown = JSON.parse(await blob.text());
    return clientAgentModelsFromExport(payload, agent);
  } catch (error: unknown) {
    if (error instanceof SyntaxError) {
      return { results: [], error: "Could not load complete model usage. Try again." };
    }
    return { results: [], error: error instanceof Error ? error.message : "Could not load the model breakdown" };
  }
};

const agentKey = (agent: AgentRow | null) =>
  agent ? [agent.id, agent.tags, agent.spend, agent.tokens, agent.requests] : null;
const timestamp = (date: Date | null) => date?.getTime() ?? Number.NaN;

export function useClientAgentModels({ accessToken, startTime, endTime, agent, enabled }: ClientAgentModelsOptions) {
  const start = timestamp(startTime);
  const end = timestamp(endTime);
  const validRange = Number.isFinite(start) && Number.isFinite(end) && start <= end;
  const active = enabled && Boolean(accessToken && agent && agent.tags.length > 0);
  const [attempt, setAttempt] = useState(0);
  const reload = useCallback(() => setAttempt((current) => current + 1), []);
  const key = JSON.stringify([attempt, accessToken, start, end, agentKey(agent)]);
  const [settled, setSettled] = useState<{ key: string; result: ClientAgentModelsResult } | null>(null);

  useEffect(() => {
    if (!active || !validRange) return;
    if (!accessToken || !agent) return;
    let cancelled = false;
    const request = { accessToken, startTime: new Date(start), endTime: new Date(end), entityIds: null };
    loadModels(request, agent).then((result) => {
      if (!cancelled) setSettled({ key, result });
    });
    return () => {
      cancelled = true;
    };
  }, [active, validRange, accessToken, start, end, agent, key]);

  const current = active && validRange && settled?.key === key ? settled.result : null;
  return {
    results: current?.results ?? [],
    loading: active && validRange && current === null,
    error: active && !validRange ? "Select a valid date range" : current?.error ?? null,
    reload,
  };
}
