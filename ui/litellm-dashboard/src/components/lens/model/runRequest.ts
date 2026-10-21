import type { RunWindow } from "./types";

export const RUN_PRESETS = [
  { label: "Since last run", hours: null },
  { label: "Last hour", hours: 1 },
  { label: "Last 24h", hours: 24 },
  { label: "Last 7d", hours: 168 },
  { label: "Custom", hours: -1 },
] as const;

export type RunPreset = (typeof RUN_PRESETS)[number]["hours"];

export interface RunChoice {
  preset: RunPreset;
  agent: string;
  saved: string;
  start: string;
  end: string;
}

export function runRequest({ preset, agent, saved, start, end }: RunChoice): RunWindow | string {
  const agentPart = agent.trim() && agent.trim() !== saved ? { agent_name: agent.trim() } : {};
  if (preset === null) return agentPart;
  if (preset > 0) return { ...agentPart, lookback_hours: preset };
  const startMs = Date.parse(start);
  const endMs = Date.parse(end);
  if (Number.isNaN(startMs) || Number.isNaN(endMs)) return "Choose a start and end time";
  if (startMs >= endMs) return "Start time must be before end time";
  return { ...agentPart, start: new Date(startMs).toISOString(), end: new Date(endMs).toISOString() };
}
