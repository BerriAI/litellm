import type { Finding, Job, Lens } from "./types";

export interface SituationInput {
  readonly lens: Lens;
  readonly job: Job | undefined;
  readonly findings: readonly Finding[] | null | undefined;
  readonly connected: boolean;
}

export type RunSituation =
  | "never"
  | "queued"
  | "running"
  | "budget"
  | "offline"
  | "failed"
  | "cancelled"
  | "unknown"
  | "issues"
  | "watching"
  | "clean";

export type RunAction = "run" | "stop" | "retry" | "raiseBudget" | "connectWorker" | "reviewIssues" | "monitor";

const failed = (job: Job | undefined) => job?.status === "failed";

export function budgetReached(lens: Lens, now = new Date()): boolean {
  const spent = lens.budget_month === now.toISOString().slice(0, 7) ? lens.spent ?? 0 : 0;
  return spent >= (lens.settings.monthly_budget ?? 100);
}

export const openIssues = (findings: readonly Finding[] | null | undefined) =>
  findings?.filter((finding) => finding.kind === "issue" && finding.status === "open").length ?? 0;

/** First match wins, so the order is the precedence between overlapping situations. */
const RULES: readonly (readonly [RunSituation, (input: SituationInput) => boolean])[] = [
  ["never", ({ job }) => !job],
  ["queued", ({ job }) => job?.status === "queued"],
  ["running", ({ job }) => job?.status === "running"],
  ["budget", ({ job, lens }) => failed(job) && (budgetReached(lens) || /budget reached/i.test(job?.error ?? ""))],
  ["offline", ({ job, connected }) => failed(job) && !connected],
  ["failed", ({ job }) => failed(job)],
  ["cancelled", ({ job }) => job?.status === "cancelled"],
  ["unknown", ({ findings }) => findings == null],
  ["issues", ({ findings }) => openIssues(findings) > 0],
  ["watching", ({ lens }) => lens.settings.enabled],
  ["clean", () => true],
];

export function runSituation(input: SituationInput): RunSituation {
  return RULES.find(([, matches]) => matches(input))?.[0] ?? "clean";
}

/** The one thing worth doing next in each situation; null when nothing needs the user. */
export const NEXT_ACTION: Record<RunSituation, RunAction | null> = {
  never: "run",
  queued: "stop",
  running: "stop",
  budget: "raiseBudget",
  offline: "connectWorker",
  failed: "retry",
  cancelled: "retry",
  unknown: null,
  issues: "reviewIssues",
  watching: null,
  clean: "monitor",
};
