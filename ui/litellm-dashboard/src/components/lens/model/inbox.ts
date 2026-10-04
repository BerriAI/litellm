import type { Finding, Job, Lens } from "./types";

export type Step = Job["steps"][number];
export type Priority = NonNullable<Finding["priority"]>;

export const UNKNOWN_AGENT = "unknown agent";

const priorityRank = { high: 0, medium: 1, low: 2 } as const;

export function sampledExecutions(lens: Lens) {
  return lens.jobs.flatMap((job) => job.sample?.executions ?? []);
}

export function findingAgents(lens: Lens, finding: Finding): readonly string[] {
  const services = new Map(sampledExecutions(lens).map((run) => [run.id, run.service] as const));
  const seen = new Set(finding.occurrences.map((id) => services.get(id)).filter((s): s is string => !!s));
  if (seen.size) return [...seen].sort();
  const configured = lens.settings.agent_name || lens.settings.service;
  return [configured || UNKNOWN_AGENT];
}

function groupBy<T>(items: readonly T[], key: (item: T) => string): Map<string, T[]> {
  return items.reduce((groups, item) => {
    const k = key(item);
    return groups.set(k, [...(groups.get(k) ?? []), item]);
  }, new Map<string, T[]>());
}

export function openFindings(lens: Lens): Finding[] {
  return lens.findings
    .filter((f) => f.status === "open" && f.kind === "issue")
    .sort(
      (a, b) =>
        priorityRank[a.priority ?? "medium"] - priorityRank[b.priority ?? "medium"] ||
        Date.parse(b.last_seen) - Date.parse(a.last_seen),
    );
}

export const findingKey = (lens: Lens, finding: Finding) => `${lens.id}:${finding.id}`;

export interface OwnedFinding {
  readonly lens: Lens;
  readonly finding: Finding;
}

export function findFinding(lenses: readonly Lens[], key: string): OwnedFinding | undefined {
  return lenses
    .flatMap((lens) => lens.findings.map((finding) => ({ lens, finding })))
    .find(({ lens, finding }) => findingKey(lens, finding) === key);
}

export function stepLine(step: Step): string {
  if (step.kind !== "model") return step.label;
  const tokens = step.prompt_tokens + step.completion_tokens;
  const size = tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k tok` : `${tokens} tok`;
  return `${step.label} · ${size} · $${step.cost.toFixed(4)}`;
}

export function modelsUsed(steps: readonly Step[]): string[] {
  const counts = groupBy(
    steps.filter((s) => s.kind === "model" && s.model),
    (s) => s.model,
  );
  return [...counts.entries()].sort((a, b) => b[1].length - a[1].length).map(([model]) => model);
}

function nextRunLabel(msUntil: number): string {
  if (msUntil <= 0) return "due now";
  if (msUntil < 60_000) return "next in <1m";
  return `next in ${Math.ceil(msUntil / 60_000)}m`;
}

export function scheduleLabel(lens: Lens, now: number): string {
  if (!lens.settings.enabled) return "paused";
  const every = lens.settings.interval_minutes;
  const cadence = every % 60 === 0 ? `every ${every / 60}h` : `every ${every}m`;
  const next = Date.parse(lens.next_run_at) - now;
  return `${cadence} · ${nextRunLabel(next)}`;
}

const WINDOW_FORMAT: Intl.DateTimeFormatOptions = {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
};

export function windowLabel(job: Job): string {
  const fmt = (iso: string) => new Date(iso).toLocaleString(undefined, WINDOW_FORMAT);
  return `${fmt(job.start)} → ${fmt(job.end)}`;
}
