import { formatActivityTimestamp } from "@/utils/activityTimestamp";
import type { Lens, LensList } from "./types";

export function workerConnected(worker: LensList["workers"][number], now = Date.now()): boolean {
  return !worker.revoked && !!worker.analysis_key_id && now - Date.parse(worker.last_seen) < 120000;
}

export function lensStatus(lens: Lens, connected: boolean): string {
  const active = lens.jobs?.find((job) => ["queued", "running"].includes(job.status ?? ""));
  if (active) return connected ? active.stage ?? "Queued" : "Waiting for analyzer";
  const spent = lens.budget_month === new Date().toISOString().slice(0, 7) ? lens.spent ?? 0 : 0;
  if (spent >= (lens.settings.monthly_budget ?? 100)) return "Budget reached";
  const latest = lens.jobs?.[0];
  if (latest?.status === "failed") return "Failed";
  if (latest?.status === "cancelled") return "Cancelled";
  if (latest?.status === "completed") return "Completed";
  return "Ready";
}

export function nextCheckStatus(lens: Lens, now: number): string | null {
  if (!lens.settings.enabled) return null;
  const active = lens.jobs.find((job) => job.status === "queued" || job.status === "running");
  if (active?.status === "running") return "Next check scheduled after this scan finishes";
  if (active?.status === "queued") return "Waiting for an analyzer";
  const next = new Date(lens.next_run_at);
  const remaining = next.getTime() - now;
  if (remaining <= 0) return "Due now · waiting for an analyzer";
  const minutes = Math.ceil(remaining / 60000);
  const relative = minutes === 1 ? "in less than a minute" : `in ${minutes} minutes`;
  const time = formatActivityTimestamp(lens.next_run_at);
  return `Next check ${time} · ${relative}`;
}

export function readiness(
  activity: { traces: boolean; requests: boolean } | undefined,
  activityError: unknown,
  connected: boolean,
  listError: unknown,
) {
  const tracesReady = activity?.traces === true && !activityError;
  const requestsReady = activity?.requests === true && !activityError;
  const activityReady = tracesReady || requestsReady;
  const ready = activityReady && connected && !listError;
  return { tracesReady, requestsReady, activityReady, ready };
}
