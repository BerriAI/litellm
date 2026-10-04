import type { Job, Review, Settings } from "./types";

export type Outcome = "issue" | "clear" | "unknown";

export interface Conclusion {
  checkId: string;
  label: string;
  latest: string;
  count: number;
  issue: boolean;
}

export interface Phase {
  span: number;
  typed: number;
  verdict: boolean;
}

export interface LiveStats {
  perSecond: number | null;
  tokens: number;
  cost: number;
  elapsedSeconds: number;
}

const PACE_WINDOW_MS = 2400;
const MIN_STEP_MS = 60;
const COMPACT_STEP_MS = 500;
const READ_SHARE = 0.35;
const TYPE_SHARE = 0.45;
const REPLAY_ON_OPEN = 3;

export function liveJob(jobs: readonly Job[]): Job | undefined {
  const active = jobs.find((job) => job.status === "queued" || job.status === "running");
  if (active) return active.reviews.length ? active : undefined;
  const latest = jobs[0];
  return latest?.reviews.length ? latest : undefined;
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

export function tickerLine(review: Pick<Review, "agent" | "name" | "trace_id">): string {
  return `reading ${review.agent || review.name} · ${review.trace_id.slice(0, 8)}`;
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

export function stripState(job: Pick<Job, "status" | "error" | "stage" | "steps" | "coverage" | "reviews">, model: string): StripState {
  if (job.status === "failed") return { kind: "failed", message: job.error || "The investigation failed" };
  const stepError = job.steps.findLast((step) => step.kind === "error");
  if (stepError && !job.reviews.length) return { kind: "failed", message: stepError.label };
  if (job.status === "completed" || job.status === "cancelled") return { kind: "done" };
  if (job.reviews.length) return { kind: "reviewing" };
  if (job.status === "queued") return { kind: "waiting", message: "Queued, waiting for a worker to pick this up" };
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

export function focusedReview(
  reviews: readonly Review[],
  pinned: string | null,
  live: Review | null,
): { review: Review | null; following: boolean } {
  const picked = pinned ? reviews.find((r) => reviewKey(r) === pinned) : undefined;
  return picked ? { review: picked, following: false } : { review: live, following: true };
}

export function verdictLine(review: Pick<Review, "cannot_assess" | "verdicts">): string {
  const issue = review.verdicts.find((v) => v.kind === "issue");
  if (issue) return issue.summary;
  if (review.cannot_assess) return "Not enough evidence to judge";
  return review.verdicts[0]?.summary ?? "No issue observed";
}

export function unseen(reviews: readonly Review[], seen: ReadonlySet<string>): Review[] {
  return reviews.filter((review) => !seen.has(reviewKey(review)));
}

export function stepDuration(backlog: number): number {
  return Math.max(MIN_STEP_MS, Math.min(PACE_WINDOW_MS, Math.round(PACE_WINDOW_MS / Math.max(1, backlog))));
}

export function isCompact(duration: number): boolean {
  return duration < COMPACT_STEP_MS;
}

export function analysisModel(candidates: readonly string[]): string {
  return candidates.find((model) => providerOf(model)) ?? candidates.find(Boolean) ?? "";
}

export function playbackPhase(elapsed: number, duration: number, spans: number, chars: number): Phase {
  if (isCompact(duration)) return { span: -1, typed: chars, verdict: true };
  const t = Math.max(0, elapsed) / duration;
  const reading = t < READ_SHARE;
  const span = reading && spans > 0 ? Math.min(spans - 1, Math.floor((t / READ_SHARE) * spans)) : -1;
  const typing = Math.min(1, Math.max(0, (t - READ_SHARE) / TYPE_SHARE));
  return { span, typed: Math.round(typing * chars), verdict: t >= READ_SHARE + TYPE_SHARE };
}

export function conclusions(reviews: readonly Review[], checks: Settings["checks"] = []): Conclusion[] {
  const instructions = new Map(checks.map((check) => [check.id, check.instruction]));
  const verdicts = reviews.flatMap((review) => review.verdicts);
  const grouped = verdicts.reduce(
    (groups, verdict) => {
      const prior = groups.get(verdict.check_id);
      return new Map(groups).set(verdict.check_id, {
        checkId: verdict.check_id,
        label: instructions.get(verdict.check_id) ?? verdict.summary,
        latest: verdict.summary,
        count: (prior?.count ?? 0) + 1,
        issue: (prior?.issue ?? false) || verdict.kind === "issue",
      });
    },
    new Map<string, Conclusion>(),
  );
  return [...grouped.values()].sort((a, b) => Number(b.issue) - Number(a.issue) || b.count - a.count);
}

export function readingStart(job: Pick<Job, "steps" | "created_at">): string {
  return job.steps.find((step) => step.kind === "stage" && step.label === "Reading executions")?.at ?? job.created_at;
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

export interface Playback {
  played: readonly Review[];
  current: Review | null;
  pending: readonly Review[];
  seen: ReadonlySet<string>;
  startedAt: number;
  duration: number;
}

export type PlaybackAction =
  | { type: "enqueue"; reviews: readonly Review[] }
  | { type: "tick"; now: number }
  | { type: "settle" }
  | { type: "replay" };

const PLAYED_LIMIT = 200;

export function startPlayback(reviews: readonly Review[], live: boolean): Playback {
  const replay = live ? Math.min(REPLAY_ON_OPEN, reviews.length) : 0;
  const shown = reviews.slice(0, reviews.length - replay);
  return {
    played: live ? shown : shown.slice(0, -1),
    current: live ? null : shown.at(-1) ?? null,
    pending: reviews.slice(shown.length),
    seen: new Set(reviews.map(reviewKey)),
    startedAt: Number.NEGATIVE_INFINITY,
    duration: 0,
  };
}

function enqueue(state: Playback, reviews: readonly Review[]): Playback {
  const fresh = unseen(reviews, state.seen);
  if (!fresh.length) return state;
  return {
    ...state,
    pending: [...state.pending, ...fresh],
    seen: new Set([...state.seen, ...fresh.map(reviewKey)]),
  };
}

function advance(state: Playback, now: number): Playback {
  const [next, ...rest] = state.pending;
  if (!next) return state;
  if (state.current && now - state.startedAt < state.duration) return state;
  return {
    ...state,
    played: state.current ? [...state.played, state.current].slice(-PLAYED_LIMIT) : state.played,
    current: next,
    pending: rest,
    startedAt: now,
    duration: stepDuration(state.pending.length),
  };
}

function settle(state: Playback): Playback {
  const all = [...state.played, ...(state.current ? [state.current] : []), ...state.pending];
  return {
    ...state,
    played: all.slice(0, -1).slice(-PLAYED_LIMIT),
    current: all.at(-1) ?? null,
    pending: [],
    startedAt: Number.NEGATIVE_INFINITY,
    duration: 0,
  };
}

function replay(state: Playback): Playback {
  const all = [...state.played, ...(state.current ? [state.current] : []), ...state.pending];
  return { ...state, played: [], current: null, pending: all, startedAt: Number.NEGATIVE_INFINITY, duration: 0 };
}

export function playbackReducer(state: Playback, action: PlaybackAction): Playback {
  switch (action.type) {
    case "enqueue":
      return enqueue(state, action.reviews);
    case "tick":
      return advance(state, action.now);
    case "settle":
      return settle(state);
    case "replay":
      return replay(state);
  }
}

export function queueRows(playback: Pick<Playback, "played" | "current">, limit: number): Review[] {
  const newest = playback.current ? [playback.current] : [];
  return [...newest, ...[...playback.played].reverse()].slice(0, limit);
}

export function shownCount(reviewed: number, playback: Pick<Playback, "played" | "current" | "pending">): number {
  const local = playback.played.length + (playback.current ? 1 : 0);
  return Math.max(local, reviewed - playback.pending.length);
}
