import type { Finding, Job, Lens } from "./types";

export type Step = Job["steps"][number];
export type Priority = NonNullable<Finding["priority"]>;

export interface InboxRow {
  key: string;
  title: string;
  agents: readonly string[];
  priority: Priority;
  suggestion: string;
  runs: number;
  lastSeen: string;
  sources: readonly { lens: Lens; finding: Finding }[];
}

export interface InboxFilter {
  agent: string;
  priority: Priority | "all";
}

export const ALL_AGENTS = "all";
export const UNKNOWN_AGENT = "unknown agent";

const priorityRank = { high: 0, medium: 1, low: 2 } as const;

export function sampledExecutions(lens: Lens) {
  return lens.jobs.flatMap((job) => job.sample?.executions ?? []);
}

export function findingAgents(lens: Lens, finding: Finding): readonly string[] {
  if (lens.settings.agent_name) return [lens.settings.agent_name];
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

function inboxGroupKey(lens: Lens, finding: Finding, agents: readonly string[]): string {
  const check = lens.settings.checks.find((candidate) => candidate.id === finding.check_id);
  return JSON.stringify([
    lens.scope.team_id,
    lens.scope.api_key_hash,
    lens.scope.all_teams,
    lens.settings.source,
    agents,
    finding.check_id,
    check?.instruction ?? lens.id,
    finding.title.trim().toLowerCase(),
  ]);
}

export function inboxRows(lenses: readonly Lens[]): InboxRow[] {
  const open = lenses.flatMap((lens) =>
    lens.findings
      .filter((f) => f.status === "open" && f.kind === "issue")
      .map((finding) => ({ lens, finding, agents: findingAgents(lens, finding) })),
  );
  const grouped = groupBy(open, ({ lens, agents, finding }) => inboxGroupKey(lens, finding, agents));
  return [...grouped.values()]
    .map((sources): InboxRow => {
      const best = sources.reduce((a, b) =>
        priorityRank[a.finding.priority ?? "medium"] <= priorityRank[b.finding.priority ?? "medium"] ? a : b,
      );
      return {
        key: findingKey(sources[0].lens, sources[0].finding),
        title: best.finding.title,
        agents: best.agents,
        priority: best.finding.priority ?? "medium",
        suggestion: best.finding.suggestion,
        runs: new Set(sources.flatMap((s) => s.finding.occurrences)).size,
        lastSeen: sources.map((s) => s.finding.last_seen).reduce((a, b) => (Date.parse(a) > Date.parse(b) ? a : b)),
        sources: sources.map(({ lens, finding }) => ({ lens, finding })),
      };
    })
    .sort(
      (a, b) => priorityRank[a.priority] - priorityRank[b.priority] || Date.parse(b.lastSeen) - Date.parse(a.lastSeen),
    );
}

export function filterInbox(rows: readonly InboxRow[], { agent, priority }: InboxFilter): InboxRow[] {
  const agentMatches = (row: InboxRow) => agent === ALL_AGENTS || row.agents.includes(agent);
  const priorityMatches = (row: InboxRow) => priority === "all" || row.priority === priority;
  return rows.filter((row) => agentMatches(row) && priorityMatches(row));
}

export function inboxAgents(rows: readonly InboxRow[]): string[] {
  return [...new Set(rows.flatMap((row) => row.agents))].sort();
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
    .find(
      ({ lens, finding }) =>
        findingKey(lens, finding) === key || finding.merged_finding_ids?.some((id) => `${lens.id}:${id}` === key),
    );
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

export const shortTime = (iso: string) => new Date(iso).toLocaleString(undefined, WINDOW_FORMAT);

export function windowLabel(job: Job): string {
  return `${shortTime(job.start)} → ${shortTime(job.end)}`;
}

export function inboxFinding(row: InboxRow): Finding {
  const primary = row.sources.find(({ finding }) => finding.priority === row.priority) ?? row.sources[0];
  const evidence = row.sources.flatMap(({ finding }) => finding.evidence);
  return {
    ...primary.finding,
    priority: row.priority,
    last_seen: row.lastSeen,
    occurrences: [...new Set(row.sources.flatMap(({ finding }) => finding.occurrences))],
    evidence: [...new Map(evidence.map((quote) => [JSON.stringify(quote), quote])).values()],
  };
}
