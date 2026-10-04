import { durationText } from "./format";
import type { Job, Review, ReviewVerdict, Settings } from "./types";

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

export interface LiveStats {
  perSecond: number | null;
  tokens: number;
  cost: number;
  elapsedSeconds: number;
}

export function liveJob(jobs: readonly Job[]): Job | undefined {
  const active = jobs.find((job) => job.status === "queued" || job.status === "running");
  if (active) return active;
  const latest = jobs[0];
  return latest && latest.reviewed > 0 ? latest : undefined;
}

export function reviewKey(review: Pick<Review, "execution_id" | "at">): string {
  return `${review.execution_id}@${review.at}`;
}

export function providerOf(model: string): string {
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
  | { kind: "reviewing" }
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
  if (job.reviews.length) return { kind: "reviewing" };
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
  if (job.reviewed > job.reviews.length) return { count, scope: `in last ${job.reviews.length} reviewed` };
  return { count, scope: "" };
}

const KEPT_REVIEWS = 200;

export interface ReviewFeed {
  reviews: readonly Review[];
  cursor: number;
}

export const EMPTY_FEED: ReviewFeed = { reviews: [], cursor: 0 };

export function appendPage(feed: ReviewFeed, page: { reviews: readonly Review[]; reviewed: number }): ReviewFeed {
  if (page.reviewed === feed.cursor && !page.reviews.length) return feed;
  const added = unseen(page.reviews, new Set(feed.reviews.map(reviewKey)));
  return { reviews: [...feed.reviews, ...added].slice(-KEPT_REVIEWS), cursor: page.reviewed };
}

export function unseen(reviews: readonly Review[], seen: ReadonlySet<string>): Review[] {
  return reviews.filter((review) => !seen.has(reviewKey(review)));
}

export function analysisModel(candidates: readonly string[]): string {
  return candidates.find((model) => providerOf(model)) ?? candidates.find(Boolean) ?? "";
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
      return new Map(next).set(verdict.check_id, {
        key: verdict.check_id,
        checkId: verdict.check_id,
        label: checkLabel(verdict.check_id, instructions.get(verdict.check_id)),
        latest: issue || !prior ? verdict.summary : prior.latest,
        count: (prior?.count ?? 0) + Number(issue),
        noted: (prior?.noted ?? 0) + Number(!issue),
        issue: (prior?.issue ?? false) || issue,
      });
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

export function liveStats(job: Job, now: number): LiveStats {
  const models = job.steps.filter((step) => step.kind === "model");
  const end = job.finished_at ? Date.parse(job.finished_at) : now;
  const elapsedSeconds = Math.max(0, Math.floor((end - Date.parse(readingStart(job))) / 1000));
  return {
    perSecond: elapsedSeconds > 0 && job.reviewed > 0 ? job.reviewed / elapsedSeconds : null,
    tokens: models.reduce((sum, step) => sum + step.prompt_tokens + step.completion_tokens, 0),
    cost: job.cost,
    elapsedSeconds,
  };
}

export function rateLabel(perSecond: number | null): string {
  if (perSecond === null) return "–";
  return perSecond >= 1 ? `${perSecond.toFixed(1)} traces/s` : `${(perSecond * 60).toFixed(1)} traces/min`;
}

export function tokenLabel(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M tok`;
  return tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k tok` : `${tokens} tok`;
}

export function newestFirst(reviews: readonly Review[], limit: number): Review[] {
  return [...reviews].reverse().slice(0, limit);
}

export interface InFlight {
  execution_id: string;
  trace_id: string;
  agent: string;
  started_at: string;
}

export function inFlight(job: Job): readonly InFlight[] {
  const reading = (job as Job & { reading?: readonly InFlight[] }).reading;
  return job.status === "running" ? (reading ?? []) : [];
}

export type LiveRow = { kind: "reading"; key: string; item: InFlight } | { kind: "done"; key: string; review: Review };

export function liveRows(reading: readonly InFlight[], reviews: readonly Review[], limit: number): LiveRow[] {
  const finished = new Set(reviews.map((review) => review.execution_id));
  const open = reading.filter((item) => !finished.has(item.execution_id));
  return [
    ...open.map((item) => ({ kind: "reading" as const, key: item.execution_id, item })),
    ...newestFirst(reviews, limit).map((review) => ({ kind: "done" as const, key: review.execution_id, review })),
  ];
}

export function nowLine(job: Pick<Job, "coverage" | "reviewed">, reading: number): string {
  const { selected } = job.coverage;
  const done = selected ? `${Math.min(job.reviewed, selected)} of ${selected} done` : `${job.reviewed} done`;
  return reading ? `Reviewing ${reading} at a time · ${done}` : done;
}

export function durationLabel(ms: number): string {
  if (ms < 1000) return `${Math.max(0, Math.round(ms))}ms`;
  return ms < 60_000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.floor(ms / 60_000)}m ${Math.round((ms % 60_000) / 1000)}s`;
}

export function doneLine(job: Pick<Job, "reviewed" | "created_at" | "finished_at" | "steps">): string {
  const end = job.finished_at ? Date.parse(job.finished_at) : Number.NaN;
  const seconds = Math.round((end - Date.parse(readingStart(job))) / 1000);
  const took = Number.isFinite(seconds) && seconds >= 0 ? ` in ${durationText(seconds)}` : "";
  return `Reviewed ${job.reviewed} ${job.reviewed === 1 ? "trace" : "traces"}${took} with`;
}
