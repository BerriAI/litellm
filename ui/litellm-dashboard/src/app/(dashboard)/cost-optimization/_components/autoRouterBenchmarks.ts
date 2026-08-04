export interface AutoRouterCacheBenchmark {
  ttl_seconds: number;
  usage_coverage_pct: number;
  hit_rate_pct: number;
  turns: number;
  hits: number;
  same_model_turns: number;
  same_model_hits: number;
  first_visit_turns: number;
  first_visit_hits: number;
  return_turns: number;
  return_hits: number;
  same_model_hit_rate_pct: number;
  first_visit_hit_rate_pct: number;
  return_hit_rate_pct: number;
  stale_miss_share_pct: number;
  warming_savable_miss_pct: number;
  warming_break_even_pct: number;
  stale_return_misses: number;
  savable_return_misses: number;
  warming_rescued_spend: number;
  warming_replay_spend: number;
  warming_net_spend: number;
}

export interface AutoRouterGroupBenchmark {
  model_group: string;
  router_kind: string;
  baseline_model: string | null;
  sessions: number;
  turns: number;
  avg_turns_per_session: number;
  avg_session_length_seconds: number;
  total_tokens: number;
  avg_tokens_per_session: number;
  actual_spend: number;
  baseline_spend: number;
  savings: number;
  savings_pct: number;
  cache: AutoRouterCacheBenchmark | null;
}

export interface AutoRouterBenchmarksResponse {
  start_date: string;
  end_date: string;
  groups: AutoRouterGroupBenchmark[];
}

export interface BenchmarkView {
  routers: number;
  sessions: number;
  turns: number;
  avg_turns_per_session: number;
  avg_session_length_seconds: number;
  avg_tokens_per_session: number;
  actual_spend: number;
  baseline_spend: number;
  savings: number;
  savings_pct: number;
  saved_per_session: number;
  baselineLabel: string | null;
  cache: AutoRouterCacheBenchmark | null;
  mixedTtl: boolean;
}

export const ALL_ROUTERS = "__all__";

const rate = (part: number, whole: number): number => (whole > 0 ? (100 * part) / whole : 0);

const sum = <T>(rows: readonly T[], pick: (row: T) => number): number =>
  rows.reduce((total, row) => total + pick(row), 0);

/**
 * Every rate is recomputed from summed counts rather than averaged across routers.
 * Averaging rates weights a router with three turns the same as one with three
 * thousand, which is how a blended hit rate ends up sitting nowhere near either.
 */
const combineCache = (caches: readonly AutoRouterCacheBenchmark[]): AutoRouterCacheBenchmark | null => {
  if (caches.length === 0) return null;

  const turns = sum(caches, (c) => c.turns);
  const hits = sum(caches, (c) => c.hits);
  const sameTurns = sum(caches, (c) => c.same_model_turns);
  const firstTurns = sum(caches, (c) => c.first_visit_turns);
  const returnTurns = sum(caches, (c) => c.return_turns);
  const returnHits = sum(caches, (c) => c.return_hits);
  const staleMisses = sum(caches, (c) => c.stale_return_misses);
  const savableMisses = sum(caches, (c) => c.savable_return_misses);
  const rescued = sum(caches, (c) => c.warming_rescued_spend);
  const replay = sum(caches, (c) => c.warming_replay_spend);

  return {
    ttl_seconds: Math.max(...caches.map((c) => c.ttl_seconds)),
    usage_coverage_pct: rate(
      sum(caches, (c) => (c.usage_coverage_pct * c.turns) / 100),
      turns,
    ),
    hit_rate_pct: rate(hits, turns),
    turns,
    hits,
    same_model_turns: sameTurns,
    same_model_hits: sum(caches, (c) => c.same_model_hits),
    first_visit_turns: firstTurns,
    first_visit_hits: sum(caches, (c) => c.first_visit_hits),
    return_turns: returnTurns,
    return_hits: returnHits,
    same_model_hit_rate_pct: rate(
      sum(caches, (c) => c.same_model_hits),
      sameTurns,
    ),
    first_visit_hit_rate_pct: rate(
      sum(caches, (c) => c.first_visit_hits),
      firstTurns,
    ),
    return_hit_rate_pct: rate(returnHits, returnTurns),
    stale_miss_share_pct: rate(staleMisses, returnTurns - returnHits),
    warming_savable_miss_pct: rate(savableMisses, turns - hits),
    warming_break_even_pct: Math.max(...caches.map((c) => c.warming_break_even_pct)),
    stale_return_misses: staleMisses,
    savable_return_misses: savableMisses,
    warming_rescued_spend: rescued,
    warming_replay_spend: replay,
    warming_net_spend: rescued - replay,
  };
};

/**
 * The baseline only reads as a single model when every router measured itself
 * against the same one; otherwise naming one of them would misattribute the rest.
 */
const combineBaselineLabel = (groups: readonly AutoRouterGroupBenchmark[]): string | null => {
  const baselines = new Set(groups.map((g) => g.baseline_model).filter((m): m is string => Boolean(m)));
  if (baselines.size === 0) return null;
  if (baselines.size === 1) return [...baselines][0];
  return "each router's own baseline";
};

export const toView = (groups: readonly AutoRouterGroupBenchmark[]): BenchmarkView | null => {
  if (groups.length === 0) return null;

  const sessions = sum(groups, (g) => g.sessions);
  if (sessions === 0) return null;

  const turns = sum(groups, (g) => g.turns);
  const actualSpend = sum(groups, (g) => g.actual_spend);
  const baselineSpend = sum(groups, (g) => g.baseline_spend);
  const savings = baselineSpend - actualSpend;
  const caches = groups.map((g) => g.cache).filter((c): c is AutoRouterCacheBenchmark => c !== null);

  return {
    routers: groups.length,
    sessions,
    turns,
    avg_turns_per_session: turns / sessions,
    avg_session_length_seconds: sum(groups, (g) => g.avg_session_length_seconds * g.sessions) / sessions,
    avg_tokens_per_session: sum(groups, (g) => g.total_tokens) / sessions,
    actual_spend: actualSpend,
    baseline_spend: baselineSpend,
    savings,
    savings_pct: baselineSpend > 0 ? (100 * savings) / baselineSpend : 0,
    saved_per_session: savings / sessions,
    baselineLabel: combineBaselineLabel(groups),
    cache: combineCache(caches),
    mixedTtl: new Set(caches.map((c) => c.ttl_seconds)).size > 1,
  };
};

export const compactNumber = (n: number): string =>
  new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(n);

export const usd = (n: number): string =>
  `${n < 0 ? "-" : ""}${new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(Math.abs(n))}`;

export const durationLabel = (seconds: number): string => {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
};

export const ttlLabel = (seconds: number): string =>
  seconds >= 3600 ? `${Math.round(seconds / 3600)}h` : `${Math.round(seconds / 60)}m`;

export const pct = (n: number, digits = 1): string => `${n.toFixed(digits)}%`;
