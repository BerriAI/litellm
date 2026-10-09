"use client";

import { parseAsString, useQueryStates } from "nuqs";
import { useCallback, useEffect } from "react";
import { useLocalStorage } from "usehooks-ts";

const SELECTED_AGENT_KEY = "litellm.lens.agent";
export const selectedAgentKey = (demo: boolean): string => (demo ? `${SELECTED_AGENT_KEY}.demo` : SELECTED_AGENT_KEY);

const AGENT_PARSERS = { agent: parseAsString.withDefault("") };

/**
 * Traces are always scoped to one agent: a shared link's agent first, then this browser's last pick if it still
 * exists, then the most recently active agent.
 */
export function resolveAgent(fromUrl: string, remembered: string, available: readonly string[]): string | null {
  if (fromUrl) return fromUrl;
  if (remembered && available.includes(remembered)) return remembered;
  return available[0] ?? null;
}

export interface AgentSelection {
  readonly agent: string | null;
  select(agent: string): void;
}

/** In the URL for sharing, and remembered per browser (separately for the sample session) across refresh and login. */
export function useAgentSelection(demo: boolean, available: readonly string[]): AgentSelection {
  const [{ agent: fromUrl }, setParams] = useQueryStates(AGENT_PARSERS, { history: "push" });
  const [remembered, setRemembered] = useLocalStorage(selectedAgentKey(demo), "", {
    serializer: (value) => value,
    deserializer: (raw) => raw,
  });
  const agent = resolveAgent(fromUrl, remembered, available);
  const implied = !fromUrl ? agent : null;
  useEffect(() => {
    if (implied) void setParams({ agent: implied }, { history: "replace" });
  }, [implied, setParams]);
  const select = useCallback(
    (next: string) => {
      setRemembered(next);
      void setParams({ agent: next });
    },
    [setParams, setRemembered],
  );
  return { agent, select };
}
