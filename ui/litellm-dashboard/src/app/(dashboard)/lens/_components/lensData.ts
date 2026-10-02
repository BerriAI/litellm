import { formatActivityTimestamp } from "@/utils/activityTimestamp";
import type { components } from "@/lib/http/schema";

export type Lens = components["schemas"]["Lens"];
export type Settings = components["schemas"]["LensSettings"];
export type LensList = components["schemas"]["LensList"];
export type Finding = components["schemas"]["Finding"];
export type Sample = components["schemas"]["Sample"];
export type WorkerCreated = components["schemas"]["WorkerCreated"];

export function workerConnected(worker: LensList["workers"][number], now = Date.now()): boolean {
  return !worker.revoked && !!worker.analysis_key_id && now - Date.parse(worker.last_seen) < 120000;
}

export function scopeLabel(settings: Partial<Pick<Settings, "service" | "agent_name" | "filters">>): string {
  return (
    [settings.agent_name, settings.service, ...(settings.filters ?? []).map((f) => `${f.key}: ${f.value}`)]
      .filter(Boolean)
      .join(" · ") || "All activity"
  );
}

export const starterQuestions = [
  "Find repeated work or tool calls that add no useful information.",
  "Find tool failures or retries that the agent does not recover from.",
  "Identify recurring user needs and successful ways the agent handles them.",
];

export function normalizeFilters(filters: NonNullable<Settings["filters"]>): Settings["filters"] {
  return filters.map((f) => {
    if (!f.key.trim() || !f.value.trim()) throw new Error("Choose a key and value for every condition, or remove it");
    return { key: f.key.trim(), value: f.value.trim() };
  });
}

export { formatActivityTimestamp as runTime } from "@/utils/activityTimestamp";

export function sortedFindings(findings: Finding[]): Finding[] {
  const rank = { high: 0, medium: 1, low: 2 };
  return [...findings].sort(
    (a, b) =>
      rank[a.priority ?? "medium"] - rank[b.priority ?? "medium"] || Date.parse(b.last_seen) - Date.parse(a.last_seen),
  );
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

export function evidenceTarget(id: string): { source: string; team: string; id: string; traceRef?: string } | null {
  try {
    const parsed: unknown = JSON.parse(atob(id.replace(/-/g, "+").replace(/_/g, "/")));
    if (!Array.isArray(parsed) || ![3, 4].includes(parsed.length) || !parsed.every((item) => typeof item === "string"))
      return null;
    return { source: parsed[0], team: parsed[1], id: parsed[2], ...(parsed[3] ? { traceRef: parsed[3] } : {}) };
  } catch {
    return null;
  }
}

export type Job = components["schemas"]["Job"];

export const analysisStages = [
  { stage: "Collecting executions", label: "Collect runs" },
  { stage: "Reading executions", label: "Review runs" },
  { stage: "Grouping observations", label: "Find patterns" },
  { stage: "Checking original evidence", label: "Check evidence" },
] as const;

export function analysisProgress(job: Job) {
  const {
    screened = 0,
    selected = 0,
    grouped_batches = 0,
    grouping_batches = 0,
    investigated = 0,
    candidates = 0,
  } = job.coverage ?? {};
  if (job.status === "queued") {
    return {
      step: -1,
      title: "Queued for your worker",
      done: 0,
      total: 0,
      detail: "The worker picks up queued investigations automatically.",
    };
  }
  if (job.stage === "Collecting executions") {
    return { step: 0, title: "Collecting runs", done: 0, total: 0, detail: "Selecting runs in the time window" };
  }
  if (job.stage === "Grouping observations") {
    return {
      step: 2,
      title: "Finding patterns",
      done: grouped_batches,
      total: grouping_batches,
      detail: grouping_batches
        ? `${grouped_batches} of ${grouping_batches} observation batches compared`
        : `Comparing observations across ${screened} reviewed runs`,
    };
  }
  if (job.stage === "Checking original evidence") {
    return {
      step: 3,
      title: "Checking evidence",
      done: investigated,
      total: candidates,
      detail: candidates
        ? `${investigated} of ${candidates} patterns checked against the original activity`
        : `${investigated} patterns checked against the original activity`,
    };
  }
  return {
    step: 1,
    title: "Reviewing activity",
    done: screened,
    total: selected,
    detail: `${screened} of ${selected} selected runs reviewed`,
  };
}

export function stageTimings(job: Job, now: number): (number | undefined)[] {
  const starts = analysisStages.map(({ stage }) => job.timeline?.find((mark) => mark.stage === stage)?.started_at);
  return starts.map((start, index) => {
    if (!start) return undefined;
    const next = starts.slice(index + 1).find(Boolean);
    return Math.max(0, ((next ? Date.parse(next) : now) - Date.parse(start)) / 1000);
  });
}

export function stageRemaining(done: number, total: number, stageSeconds: number | undefined): number | undefined {
  if (stageSeconds === undefined || done < 1 || total <= done) return undefined;
  return ((total - done) * stageSeconds) / done;
}

export function formatDuration(totalSeconds: number): string {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  if (!Number.isFinite(seconds)) return "0s";
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

export function remainingLabel(seconds: number): string {
  if (seconds < 60) return "Less than a minute left";
  return `About ${Math.ceil(seconds / 60)} min left`;
}

export function analysisElapsed(createdAt: string, now: number): string {
  return formatDuration((now - Date.parse(createdAt)) / 1000);
}

export interface AnalysisModelInfo {
  model_group: string;
  providers: string[];
  mode?: string | null;
  supported_openai_params?: string[] | null;
}

export function analysisModelOptions(models: string[], details: AnalysisModelInfo[]) {
  return [...new Set(models)].sort().map((name) => {
    const info = details.find((item) => item.model_group === name);
    const capability = () => {
      if (info?.mode && info.mode !== "chat") return `${info.mode}: not suitable for Lens`;
      if (info?.supported_openai_params?.includes("response_format")) return "JSON output supported";
      return "JSON output support unverified";
    };
    return {
      value: name,
      label: name,
      sublabel: [info?.providers.join(", "), capability()].filter(Boolean).join(" · "),
    };
  });
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
