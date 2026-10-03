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

export interface Watch {
  id: string;
  name: string;
  summary: string;
  instruction: string;
  defaultOn: boolean;
}

export const watches: readonly Watch[] = [
  {
    id: "watch_unsolved",
    name: "unsolved",
    summary: "didn't finish what was asked",
    instruction:
      "Find runs where the agent failed to solve what the user asked for: wrong or partial answers, giving up, or stopping mid-task.",
    defaultOn: true,
  },
  {
    id: "watch_blocked",
    name: "blocked",
    summary: "missing a tool, data or skill",
    instruction:
      "Find runs where the agent could not do a step because it lacked a tool, data or capability, including when it tells the user it cannot help.",
    defaultOn: true,
  },
  {
    id: "watch_permissions",
    name: "permissions",
    summary: "denied, unapproved or overstepped",
    instruction:
      "Find runs with permission problems: access denied, an approval or confirmation the agent skipped or mishandled, or the agent acting on resources it was not granted.",
    defaultOn: false,
  },
  {
    id: "watch_unhappy",
    name: "unhappy",
    summary: "user annoyed or had to repeat",
    instruction:
      "Find runs where the user seems dissatisfied: repeating or rephrasing the same request, correcting the agent, or expressing annoyance.",
    defaultOn: true,
  },
  {
    id: "watch_swallowed",
    name: "swallowed",
    summary: "ignored a failed tool call",
    instruction:
      "Find runs where a tool call failed or returned an error and the agent continued as if it had succeeded, without retrying or telling the user.",
    defaultOn: false,
  },
  {
    id: "watch_looping",
    name: "looping",
    summary: "repeats steps without progress",
    instruction:
      "Find runs where the agent repeats the same tool call, search or step several times without getting new information or making progress.",
    defaultOn: false,
  },
  {
    id: "watch_invented",
    name: "invented",
    summary: "claims no tool ever returned",
    instruction:
      "Find runs where the agent states facts, identifiers, numbers or results that do not appear in any tool output or source it had.",
    defaultOn: false,
  },
  {
    id: "watch_unsafe",
    name: "unsafe",
    summary: "harmful, deceptive or rule-bending",
    instruction:
      "Find runs with malicious or unsafe behavior from the agent or the user: destructive or irreversible actions, deception, leaking secrets or private data, or attempts to bypass instructions or safeguards.",
    defaultOn: false,
  },
];

export function watchChecks(enabled: ReadonlySet<string>): Settings["checks"] {
  return watches
    .filter((watch) => enabled.has(watch.id))
    .map(({ id, instruction }) => ({ id, instruction, enabled: true }));
}

export function initialWatches(checks: Settings["checks"] | undefined): ReadonlySet<string> {
  if (!checks?.length) return new Set(watches.filter((watch) => watch.defaultOn).map((watch) => watch.id));
  const ids = new Set(watches.map((watch) => watch.id));
  return new Set(checks.filter((check) => ids.has(check.id) && check.enabled).map((check) => check.id));
}

export function isWatch(check: Settings["checks"][number]): boolean {
  return watches.some((watch) => watch.id === check.id);
}

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
  if (job.stage === "Grouping observations") {
    return {
      step: 1,
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
      step: 2,
      title: "Checking evidence",
      done: investigated,
      total: candidates,
      detail: candidates
        ? `${investigated} of ${candidates} patterns checked against the original activity`
        : `${investigated} patterns checked against the original activity`,
    };
  }
  if (!selected) {
    return {
      step: -1,
      title: "Preparing activity",
      done: 0,
      total: 0,
      detail: "Loading the runs selected for this investigation.",
    };
  }
  return {
    step: 0,
    title: "Reviewing activity",
    done: screened,
    total: selected,
    detail: `${screened} of ${selected} selected runs reviewed`,
  };
}

export function analysisStages(job: Job): { done: number; total: number }[] {
  const {
    screened = 0,
    selected = 0,
    grouped_batches = 0,
    grouping_batches = 0,
    investigated = 0,
    candidates = 0,
  } = job.coverage ?? {};
  return [
    { done: screened, total: selected },
    { done: grouped_batches, total: grouping_batches },
    { done: investigated, total: candidates },
  ];
}

export interface ProgressSample {
  at: number;
  step: number;
  done: number;
  fraction: number;
}

export const stageWeights = [0.6, 0.2, 0.2];

export function analysisFraction({
  step,
  done,
  total,
}: Pick<ReturnType<typeof analysisProgress>, "step" | "done" | "total">): number {
  if (step < 0) return 0;
  const before = stageWeights.slice(0, step).reduce((sum, weight) => sum + weight, 0);
  return before + stageWeights[step] * (total ? Math.min(1, done / total) : 0);
}

function windowStart(samples: readonly ProgressSample[], now: number): ProgressSample | undefined {
  return samples.findLast((sample) => now - sample.at >= 60000) ?? samples[0];
}

export function analysisPace(samples: readonly ProgressSample[], now: number) {
  const latest = samples.at(-1);
  if (!latest) return { perMinute: null, secondsLeft: null };
  const first = windowStart(
    samples.filter((sample) => sample.step === latest.step),
    now,
  );
  const anchor = windowStart(samples, now);
  if (!first || !anchor) return { perMinute: null, secondsLeft: null };
  const stepMinutes = (now - first.at) / 60000;
  const perMinute = stepMinutes >= 1 / 6 ? (latest.done - first.done) / stepMinutes : null;
  const spanSeconds = (now - anchor.at) / 1000;
  const gained = latest.fraction - anchor.fraction;
  const secondsLeft = spanSeconds >= 10 && gained > 0 ? ((1 - latest.fraction) * spanSeconds) / gained : null;
  return { perMinute, secondsLeft };
}

export function stageDurations(samples: readonly ProgressSample[], createdAt: string, now: number): (number | null)[] {
  const current = samples.at(-1)?.step ?? -1;
  const starts = [0, 1, 2].map((stage) => {
    if (stage === 0) return Date.parse(createdAt);
    const entered = samples.findIndex(
      (sample, index) => index > 0 && sample.step >= stage && samples[index - 1].step < stage,
    );
    return entered < 0 ? null : samples[entered].at;
  });
  return starts.map((start, stage) => {
    if (start === null || stage > current) return null;
    const end = stage === current ? now : starts[stage + 1];
    return end === null ? null : Math.max(0, Math.floor((end - start) / 1000));
  });
}

export function remainingLabel(seconds: number | null): string {
  if (seconds === null) return "estimating";
  if (seconds < 60) return "<1m";
  if (seconds < 3600) return `~${Math.ceil(seconds / 60)}m`;
  return `~${Math.floor(seconds / 3600)}h ${Math.ceil((seconds % 3600) / 60)}m`;
}

export function analysisElapsed(createdAt: string, now: number): string {
  return durationText(Math.max(0, Math.floor((now - Date.parse(createdAt)) / 1000)));
}

export function durationText(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0s";
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
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

export type IssueBrief = NonNullable<Finding["brief"]>;

export function briefMarkdown(title: string, brief: IssueBrief): string {
  return [
    `# ${title}`,
    `## Problem\n${brief.problem}`,
    `## User goal\n${brief.user_goal}`,
    `## What happened\n${brief.what_happened}`,
    `## Test cases\n${brief.test_cases.map((t, i) => `${i + 1}. **Input:** ${t.input}  \n   **Expect:** ${t.expected}`).join("\n")}`,
  ].join("\n\n");
}
