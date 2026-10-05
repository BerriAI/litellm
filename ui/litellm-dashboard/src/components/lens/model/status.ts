import { formatActivityTimestamp } from "@/utils/activityTimestamp";
import type { Job, Lens, LensList } from "./types";

export function workerConnected(worker: LensList["workers"][number], now = Date.now()): boolean {
  return !worker.revoked && !!worker.analysis_key_id && now - Date.parse(worker.last_seen) < 120000;
}

export function readingStart(job: Pick<Job, "steps" | "created_at">): string {
  return job.steps.find((step) => step.kind === "stage" && step.label === "Reading executions")?.at ?? job.created_at;
}

export function secondsToFinishReading(
  job: Pick<Job, "steps" | "created_at" | "reviewed" | "coverage">,
  now: number,
): number | null {
  const remaining = job.coverage.selected - job.reviewed;
  const elapsed = (now - Date.parse(readingStart(job))) / 1000;
  if (remaining <= 0 || job.reviewed <= 0 || elapsed <= 0) return null;
  return Math.ceil(remaining / (job.reviewed / elapsed));
}

const PICKUP_GRACE_SECONDS = 6;

export interface WorkerTask {
  lensId: string;
  name: string;
  stage: string;
  reviewed: number;
  selected: number;
  secondsLeft: number | null;
}

export type QueueReason =
  | { kind: "busy"; tasks: readonly WorkerTask[]; startsIn: number | null }
  | { kind: "no_worker" }
  | { kind: "starting"; seconds: number; tasks: readonly WorkerTask[] };

type QueuedJob = Pick<Job, "id" | "worker_id" | "created_at">;

function workerTasks(
  job: QueuedJob,
  lenses: readonly Pick<Lens, "id" | "jobs" | "settings">[],
  workerIds: ReadonlySet<string>,
  now: number,
) {
  return lenses.flatMap((lens) =>
    lens.jobs
      .filter((other) => other.status === "running" && other.id !== job.id && workerIds.has(other.worker_id ?? ""))
      .map((other) => ({
        workerId: other.worker_id ?? "",
        task: {
          lensId: lens.id,
          name: lens.settings.name,
          stage: other.stage,
          reviewed: other.reviewed,
          selected: other.coverage.selected,
          secondsLeft: secondsToFinishReading(other, now),
        },
      })),
  );
}

export function queueReason(
  job: QueuedJob,
  lenses: readonly Pick<Lens, "id" | "jobs" | "settings">[],
  workers: readonly LensList["workers"][number][],
  now: number,
): QueueReason {
  const connected = workers.filter((worker) => workerConnected(worker, now)).map((worker) => worker.id);
  if (!connected.length) return { kind: "no_worker" };
  const eligible = new Set(job.worker_id ? [job.worker_id] : connected);
  const running = workerTasks(job, lenses, eligible, now);
  const tasks = running.map((entry) => entry.task);
  const seconds = Math.max(0, Math.floor((now - Date.parse(job.created_at)) / 1000));
  if (!tasks.length || seconds < PICKUP_GRACE_SECONDS) return { kind: "starting", seconds, tasks };
  const estimates = tasks.flatMap((task) => (task.secondsLeft === null ? [] : [task.secondsLeft]));
  return { kind: "busy", tasks, startsIn: estimates.length ? Math.min(...estimates) : null };
}

function waitLabel(seconds: number): string {
  if (seconds < 60) return `~${Math.max(5, Math.ceil(seconds / 5) * 5)}s`;
  return `~${Math.ceil(seconds / 60)}m`;
}

export function queueReasonText(reason: QueueReason): string {
  switch (reason.kind) {
    case "busy": {
      const count = reason.tasks.length;
      const when = reason.startsIn === null ? "" : ` · starts in ${waitLabel(reason.startsIn)}`;
      return `Worker is busy with ${count} ${count === 1 ? "investigation" : "investigations"}${when}`;
    }
    case "no_worker":
      return "No worker connected. Start one from Connect worker.";
    case "starting":
      return `Picking up… ${reason.seconds}s`;
  }
}

export function workerTaskText(task: WorkerTask): string {
  const progress = task.selected ? ` · ${task.reviewed} of ${task.selected} traces` : "";
  return `${task.name} · ${task.stage || "starting"}${progress}`;
}

export function isActive(job: Pick<Job, "status">): boolean {
  return job.status === "queued" || job.status === "running";
}

export function activeJob(jobs: readonly Job[]): Job | undefined {
  return jobs.find(isActive);
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
