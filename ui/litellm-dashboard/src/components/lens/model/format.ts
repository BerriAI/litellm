import { formatActivityTimestamp as runTime } from "@/utils/activityTimestamp";
import type { Settings } from "./types";

export { runTime };

export function scopeLabel(settings: Partial<Pick<Settings, "service" | "agent_name" | "filters">>): string {
  return (
    [settings.agent_name, settings.service, ...(settings.filters ?? []).map((f) => `${f.key}: ${f.value}`)]
      .filter(Boolean)
      .join(" · ") || "All activity"
  );
}

export function durationText(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0s";
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

export function durationLabel(value: number, base: "minutes" | "hours" = "minutes"): string {
  const minutes = base === "hours" ? value * 60 : value;
  if (minutes >= 1440) {
    const days = Number((minutes / 1440).toFixed(2));
    return `${days} ${days === 1 ? "day" : "days"}`;
  }
  if (minutes % 60 === 0) return `${minutes / 60} ${minutes === 60 ? "hour" : "hours"}`;
  return `${minutes} ${minutes === 1 ? "minute" : "minutes"}`;
}

export function agoLabel(thenMs: number, nowMs: number): string {
  const seconds = Math.max(0, Math.floor((nowMs - thenMs) / 1000));
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export const money = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 3 }).format(n);
export const when = (value?: string | null) => (value ? runTime(value) : "Not yet");

export const sourceLabels = { both: "Traces and requests", requests: "LLM requests", traces: "Agent traces" };
