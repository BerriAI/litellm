import { formatActivityTimestamp } from "@/utils/activityTimestamp";
import type { Job, Lens, LensList } from "./types";

export function workerConnected(worker: LensList["workers"][number], now = Date.now()): boolean {
  return !worker.revoked && !!worker.analysis_key_id && now - Date.parse(worker.last_seen) < 120000;
}

export function activeJob(jobs: readonly Job[]): Job | undefined {
  return jobs.find((job) => job.status === "queued" || job.status === "running");
}

export function hasActiveJob(jobs: readonly Job[]): boolean {
  return activeJob(jobs) !== undefined;
}

/** One observer polls `/lens`: fast while work is in flight or a worker is being connected, slow otherwise. */
export function listPollInterval(list: LensList | undefined, settingsOpen: boolean, now: number): number {
  const running = list?.lenses.some((lens) => hasActiveJob(lens.jobs)) ?? false;
  const connected = list?.workers.some((worker) => workerConnected(worker, now)) ?? false;
  return running || (settingsOpen && !connected) ? 2000 : 10000;
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

export type InvestigationActivity = "running" | "queued" | "idle";

export function investigationActivity(lenses: readonly Lens[]): InvestigationActivity {
  const statuses = new Set(lenses.flatMap((lens) => lens.jobs.map((job) => job.status)));
  if (statuses.has("running")) return "running";
  if (statuses.has("queued")) return "queued";
  return "idle";
}
