import type { InFlight } from "./types";
import type { Review } from "./types";

export interface Lane {
  key: string;
  agent: string;
  traceId: string;
  startedAt: number;
  review: Review | null;
  landedAt: number | null;
  goneAt: number | null;
}

export interface Stage {
  lanes: readonly Lane[];
  released: ReadonlySet<string>;
}

export interface StageInput {
  reading: readonly InFlight[];
  reviews: readonly Review[];
  now: number;
  slots: number;
  running: boolean;
  charMs: number;
}

const VERDICT_HOLD_MS = 1200;
const GONE_MS = 4000;
const LANE_CHARS = 220;
const LANE_SENTENCES = 2;

export function laneText(reasoning: string): string {
  const brief = reasoning
    .trim()
    .split(/(?<=[.!?])\s+/)
    .slice(0, LANE_SENTENCES)
    .join(" ");
  return brief.length > LANE_CHARS ? `${brief.slice(0, LANE_CHARS - 1).trimEnd()}…` : brief;
}

export function holdMs(review: Pick<Review, "reasoning">, charMs: number): number {
  return laneText(review.reasoning).length * charMs + VERDICT_HOLD_MS;
}

export function typedChars(lane: Pick<Lane, "landedAt">, now: number, charMs: number): number {
  if (lane.landedAt === null) return 0;
  return charMs > 0 ? Math.floor((now - lane.landedAt) / charMs) : Number.MAX_SAFE_INTEGER;
}

export function startStage(reviews: readonly Review[]): Stage {
  return { lanes: [], released: new Set(reviews.map((review) => review.execution_id)) };
}

function settle(lane: Lane, reviews: ReadonlyMap<string, Review>, live: ReadonlySet<string>, now: number): Lane {
  const review = lane.review ?? reviews.get(lane.key) ?? null;
  const landedAt = lane.landedAt ?? (review ? now : null);
  const goneAt = review || live.has(lane.key) ? null : lane.goneAt ?? now;
  if (review === lane.review && landedAt === lane.landedAt && goneAt === lane.goneAt) return lane;
  return { ...lane, review, landedAt, goneAt };
}

function expired(lane: Lane, now: number, charMs: number): boolean {
  if (lane.review && lane.landedAt !== null) return now - lane.landedAt >= holdMs(lane.review, charMs);
  return lane.goneAt !== null && now - lane.goneAt >= GONE_MS;
}

function landedLane(review: Review, now: number): Lane {
  return {
    key: review.execution_id,
    agent: review.agent || review.name,
    traceId: review.trace_id,
    startedAt: now,
    review,
    landedAt: now,
    goneAt: null,
  };
}

function readingLane(item: InFlight): Lane {
  return {
    key: item.execution_id,
    agent: item.agent,
    traceId: item.trace_id,
    startedAt: Date.parse(item.started_at),
    review: null,
    landedAt: null,
    goneAt: null,
  };
}

export function stepStage(stage: Stage, input: StageInput): Stage {
  const { now, charMs } = input;
  if (!input.running) {
    if (!stage.lanes.length && input.reviews.every((r) => stage.released.has(r.execution_id))) return stage;
    return { lanes: [], released: new Set([...stage.released, ...input.reviews.map((r) => r.execution_id)]) };
  }
  const reviews = new Map(input.reviews.map((review) => [review.execution_id, review]));
  const live = new Set(input.reading.map((item) => item.execution_id));
  const settled = stage.lanes.map((lane) => settle(lane, reviews, live, now));
  const done = settled.filter((lane) => expired(lane, now, charMs));
  const kept = settled.filter((lane) => !done.includes(lane));
  const taken = new Set([...settled.map((lane) => lane.key), ...stage.released]);
  const fresh = input.reviews.filter((review) => !taken.has(review.execution_id));
  const free = Math.max(0, input.slots - kept.length);
  const landing = free ? fresh.slice(-free) : [];
  const overflow = fresh.slice(0, fresh.length - landing.length);
  const room = free - landing.length;
  const waiting = input.reading.filter((item) => !taken.has(item.execution_id) && !reviews.has(item.execution_id));
  const starting = room > 0 ? waiting.slice(0, room) : [];
  const nothingMoved = !done.length && !fresh.length && !starting.length;
  if (nothingMoved && settled.every((lane, n) => lane === stage.lanes[n])) return stage;
  const releasedNow = [...done.filter((lane) => lane.review), ...overflow].map((item) =>
    "key" in item ? item.key : item.execution_id,
  );
  return {
    lanes: [...kept, ...landing.map((review) => landedLane(review, now)), ...starting.map(readingLane)],
    released: releasedNow.length ? new Set([...stage.released, ...releasedNow]) : stage.released,
  };
}

const TYPING_TICK_MS = 30;
const CLOCK_TICK_MS = 500;

export function stageTick(stage: Stage, running: boolean, now: number, charMs: number): number | null {
  const typing = stage.lanes.some(
    (lane) => lane.review && typedChars(lane, now, charMs) < laneText(lane.review.reasoning).length,
  );
  if (typing) return TYPING_TICK_MS;
  return running || stage.lanes.length ? CLOCK_TICK_MS : null;
}

export function releasedReviews(reviews: readonly Review[], stage: Stage): Review[] {
  return reviews.filter((review) => stage.released.has(review.execution_id));
}
