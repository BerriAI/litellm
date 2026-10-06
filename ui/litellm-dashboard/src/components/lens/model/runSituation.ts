import { isPartial } from "./status";
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
  | "partial"
  | "unknown"
  | "issues"
  | "watching"
  | "clean";

export type RunAction = "run" | "stop" | "retry" | "raiseBudget" | "connectWorker" | "reviewIssues" | "monitor";

const failed = (job: Job | undefined) => job?.status === "failed";

export const openIssues = (findings: readonly Finding[] | null | undefined) =>
  findings?.filter((finding) => finding.kind === "issue" && finding.status === "open").length ?? 0;

// First match wins, so the order is the precedence between overlapping situations
const RULES: readonly (readonly [RunSituation, (input: SituationInput) => boolean])[] = [
  ["never", ({ job }) => !job],
  ["queued", ({ job }) => job?.status === "queued"],
  ["running", ({ job }) => job?.status === "running"],
  [
    "budget",
    ({ job }) =>
      job !== undefined &&
      (failed(job) || isPartial(job)) &&
      /monthly lens budget reached|remains in the investigation budget/i.test(job.error),
  ],
  ["offline", ({ job, connected }) => failed(job) && !connected],
  ["failed", ({ job }) => failed(job)],
  ["cancelled", ({ job }) => job?.status === "cancelled"],
  ["partial", ({ job }) => job !== undefined && isPartial(job)],
  ["unknown", ({ findings }) => findings == null],
  ["issues", ({ findings }) => openIssues(findings) > 0],
  ["watching", ({ lens }) => lens.settings.enabled],
  ["clean", () => true],
];

export function runSituation(input: SituationInput): RunSituation {
  return RULES.find(([, matches]) => matches(input))?.[0] ?? "clean";
}

export const NEXT_ACTION: Record<RunSituation, RunAction | null> = {
  never: "run",
  queued: "stop",
  running: "stop",
  budget: "raiseBudget",
  offline: "connectWorker",
  failed: "retry",
  cancelled: "retry",
  partial: "retry",
  unknown: null,
  issues: "reviewIssues",
  watching: null,
  clean: "monitor",
};
