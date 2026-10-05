import { durationText } from "./format";
import { activeJob, isActive, readingStart } from "./status";
import type { Activity, InFlight, Job, Review, ReviewVerdict, Settings, ToolCount } from "./types";

export type Outcome = "issue" | "clear" | "unknown";

export interface Conclusion {
  key: string;
  checkId: string;
  label: string;
  latest: string;
  count: number;
  noted: number;
  issue: boolean;
}

export function liveJob(jobs: readonly Job[]): Job | undefined {
  const active = activeJob(jobs);
  if (active) return active;
  const latest = jobs[0];
  return latest && latest.reviewed > 0 ? latest : undefined;
}

export function reviewKey(review: Pick<Review, "execution_id" | "at">): string {
  return `${review.execution_id}@${review.at}`;
}

export type ProviderCatalog = Readonly<Record<string, { litellm_provider?: string } | undefined>>;

export function providerOf(model: string, catalog: ProviderCatalog = {}): string {
  const known = catalog[model]?.litellm_provider;
  if (known) return known.toLowerCase();
  const slash = model.indexOf("/");
  return slash > 0 ? model.slice(0, slash).toLowerCase() : "";
}

export function outcome(review: Pick<Review, "cannot_assess" | "verdicts">): Outcome {
  if (review.cannot_assess) return "unknown";
  return review.verdicts.some((v) => v.kind === "issue") ? "issue" : "clear";
}

export function shortVerdict(review: Pick<Review, "cannot_assess" | "verdicts">): string {
  const issue = review.verdicts.find((v) => v.kind === "issue");
  if (issue) return issue.summary;
  return review.cannot_assess ? "not enough evidence" : "no issues";
}

export type StripState =
  | { kind: "failed"; message: string }
  | { kind: "waiting"; message: string }
  | { kind: "reviewing"; message: string }
  | { kind: "done" };

export function stripState(
  job: Pick<Job, "status" | "error" | "stage" | "steps" | "coverage" | "reviews">,
  model: string,
  queued = "Queued, waiting for a worker to pick this up",
): StripState {
  if (job.status === "failed") return { kind: "failed", message: job.error || "The investigation failed" };
  const stepError = job.steps.findLast((step) => step.kind === "error");
  if (stepError && !job.reviews.length) return { kind: "failed", message: stepError.label };
  if (job.status === "completed" || job.status === "cancelled") return { kind: "done" };
  if (job.reviews.length) return { kind: "reviewing", message: job.stage || "Reviewing traces" };
  if (job.status === "queued") return { kind: "waiting", message: queued };
  const { selected } = job.coverage;
  const using = model ? ` with ${model}` : "";
  if (job.stage === "Reading executions" && selected) {
    return { kind: "waiting", message: `Reading ${selected} ${selected === 1 ? "trace" : "traces"}${using}…` };
  }
  return { kind: "waiting", message: `${job.stage || "Starting"}${using}…` };
}

export interface IssueCount {
  count: number;
  scope: string;
}

export function issueCount(job: Pick<Job, "status" | "findings" | "reviews" | "reviewed">): IssueCount {
  const findings = job.findings?.filter((f) => f.kind === "issue").length;
  if (job.status === "completed" && findings !== undefined) return { count: findings, scope: "findings" };
  const count = job.reviews.filter((r) => outcome(r) === "issue").length;
  if (job.reviewed > job.reviews.length) return { count, scope: `in ${job.reviews.length} displayed reviews` };
  return { count, scope: "" };
}

const KEPT_REVIEWS = 200;

export interface ReviewFeed {
  reviews: readonly Review[];
  cursor: number;
}

export const EMPTY_FEED: ReviewFeed = { reviews: [], cursor: 0 };

export function appendPage(feed: ReviewFeed, page: { reviews: readonly Review[]; reviewed: number }): ReviewFeed {
  const added = unseen(page.reviews, new Set(feed.reviews.map(reviewKey)));
  if (page.reviewed === feed.cursor && !added.length) return feed;
  return { reviews: [...feed.reviews, ...added].slice(-KEPT_REVIEWS), cursor: page.reviewed };
}

export function polling(job: Pick<Job, "status" | "reviewed">, feed: ReviewFeed): boolean {
  return isActive(job) || feed.cursor < job.reviewed;
}

export function unseen(reviews: readonly Review[], seen: ReadonlySet<string>): Review[] {
  return reviews.filter((review) => !seen.has(reviewKey(review)));
}

export function analysisModel(candidates: readonly string[], catalog: ProviderCatalog = {}): string {
  return candidates.find((model) => providerOf(model, catalog)) ?? candidates.find(Boolean) ?? "";
}

const SHORT_LABEL = 48;

function humanize(checkId: string): string {
  const words = checkId.replace(/[_-]+/g, " ").trim();
  return words ? words[0].toUpperCase() + words.slice(1) : checkId;
}

export function checkLabel(checkId: string, instruction: string | undefined): string {
  return instruction && instruction.length <= SHORT_LABEL ? instruction : humanize(checkId);
}

function verdictsByCheck(review: Pick<Review, "verdicts">): Map<string, ReviewVerdict> {
  const ranked = [...review.verdicts].sort((a, b) => Number(a.kind === "issue") - Number(b.kind === "issue"));
  return new Map(ranked.map((verdict) => [verdict.check_id, verdict]));
}

export function conclusions(reviews: readonly Review[], checks: Settings["checks"] = []): Conclusion[] {
  const instructions = new Map(checks.map((check) => [check.id, check.instruction]));
  const grouped = reviews.reduce((groups, review) => {
    return [...verdictsByCheck(review).values()].reduce((next, verdict) => {
      const prior = next.get(verdict.check_id);
      const issue = verdict.kind === "issue";
      const merged: Conclusion = {
        key: verdict.check_id,
        checkId: verdict.check_id,
        label: checkLabel(verdict.check_id, instructions.get(verdict.check_id)),
        latest: issue || !prior ? verdict.summary : prior.latest,
        count: (prior?.count ?? 0) + Number(issue),
        noted: (prior?.noted ?? 0) + Number(!issue),
        issue: (prior?.issue ?? false) || issue,
      };
      return new Map(next).set(verdict.check_id, merged);
    }, groups);
  }, new Map<string, Conclusion>());
  return [...grouped.values()].sort((a, b) => b.count - a.count || b.noted - a.noted);
}

export function share(count: number, total: number): number {
  return total > 0 ? Math.min(1, count / total) : 0;
}

export function inGroup(review: Pick<Review, "verdicts">, checkId: string | null): boolean {
  return checkId === null || review.verdicts.some((verdict) => verdict.check_id === checkId);
}

const BRIEF_SENTENCES = 3;

export function briefReasoning(reasoning: string): string {
  const sentences = reasoning.trim().split(/(?<=[.!?])\s+/);
  return sentences.slice(0, BRIEF_SENTENCES).join(" ").trim();
}

export function grownGroups(before: readonly Conclusion[], after: readonly Conclusion[]): Set<string> {
  const prior = new Map(before.map((group) => [group.key, group.count + group.noted]));
  return new Set(
    after.filter((group) => group.count + group.noted > (prior.get(group.key) ?? 0)).map((group) => group.key),
  );
}

export function newestFirst(reviews: readonly Review[], limit: number): Review[] {
  return [...reviews].reverse().slice(0, limit);
}

export function inFlight(job: Pick<Job, "status" | "reading">): readonly InFlight[] {
  return job.status === "running" ? job.reading ?? [] : [];
}

export function activeActivities(job: Pick<Job, "status" | "activities">): readonly Activity[] {
  return job.status === "running" ? (job.activities ?? []).filter((activity) => !activity.finished) : [];
}

const PHASE_LABELS: Record<Activity["phase"], string> = {
  load: "Loading trace evidence",
  review: "Reviewing trace",
  group: "Grouping observations",
  reconcile: "Combining candidate groups",
  investigate: "Investigating candidate",
};

const TOOL_LABELS: Record<ToolCount["name"], string> = {
  model: "Model",
  read: "Read",
  search: "Search",
  python: "Python",
  catalog: "Trace catalog",
  review_catalog: "Review catalog",
  read_reviews: "Read reviews",
  search_reviews: "Search reviews",
  history: "History",
  checkpoint: "Context compaction",
};

const OPERATION_LABELS: Record<ToolCount["name"], string> = {
  model: "Analyzing evidence",
  read: "Reading trace content",
  search: "Searching trace content",
  python: "Running Python",
  catalog: "Inspecting trace catalog",
  review_catalog: "Inspecting review catalog",
  read_reviews: "Reading review observations",
  search_reviews: "Searching review observations",
  history: "Reading prior analysis",
  checkpoint: "Compacting context",
};

export function activityPhase(activity: Pick<Activity, "phase">): string {
  return PHASE_LABELS[activity.phase];
}

export function activityOperation(activity: Pick<Activity, "phase" | "operations">): string {
  const operations = activity.operations ?? [];
  return operations.length
    ? operations.map((operation) => OPERATION_LABELS[operation]).join(" · ")
    : activityPhase(activity);
}

export function toolCallSummary(calls: readonly ToolCount[] = []): string {
  return calls
    .filter((tool) => tool.calls > 0)
    .map((tool) => `${TOOL_LABELS[tool.name]} × ${tool.calls}`)
    .join(" · ");
}

export function reviewScope(reviewed: number, displayed: number): string {
  return reviewed > displayed ? `${displayed} displayed of ${reviewed} reviewed` : "";
}

export function nowLine(job: Pick<Job, "coverage" | "reviewed">, reading: number): string {
  const { selected } = job.coverage;
  const done = selected ? `${Math.min(job.reviewed, selected)} of ${selected}` : `${job.reviewed} done`;
  return reading ? `${done} · ${reading} in flight` : done;
}

export function durationLabel(ms: number): string {
  if (ms < 1000) return `${Math.max(0, Math.round(ms))}ms`;
  return ms < 60_000
    ? `${(ms / 1000).toFixed(1)}s`
    : `${Math.floor(ms / 60_000)}m ${Math.round((ms % 60_000) / 1000)}s`;
}

export function doneLine(job: Pick<Job, "reviewed" | "created_at" | "finished_at" | "steps">): string {
  const end = job.finished_at ? Date.parse(job.finished_at) : Number.NaN;
  const seconds = Math.round((end - Date.parse(readingStart(job))) / 1000);
  const took = Number.isFinite(seconds) && seconds >= 0 ? ` in ${durationText(seconds)}` : "";
  return `Reviewed ${job.reviewed} ${job.reviewed === 1 ? "trace" : "traces"}${took} with`;
}
