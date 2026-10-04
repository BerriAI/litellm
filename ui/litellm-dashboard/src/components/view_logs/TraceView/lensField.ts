export const AGENT_DOT_COLORS = ["#0011b3", "#3b5bfd", "#7c93ff", "#22b3e8"] as const;
export const FIELD_ROWS = 12;
export const FIELD_COLS = 2;

export type FieldDot = { kind: "run"; color: string } | { kind: "failed" } | { kind: "grid" };

export interface FieldBucket {
  runs: number;
  failed: number;
  agents: readonly string[];
}

const hash = (seed: number, value: number): number => Math.imul(seed ^ Math.imul(value, 0x9e3779b1), 0x85ebca6b) >>> 0;

export const UNNAMED_AGENT_COLOR = "#94a3b8";

export function agentDotColor(agent: string): string {
  if (!agent) return UNNAMED_AGENT_COLOR;
  const code = Array.from(agent).reduce((total, char) => hash(total, char.charCodeAt(0)), 7);
  return AGENT_DOT_COLORS[code % AGENT_DOT_COLORS.length];
}

/** Lit dot count for a bucket: square-root scaled against the busiest bucket so sparse traffic still reads. */
export function litDots(runs: number, max: number, capacity = FIELD_ROWS * FIELD_COLS): number {
  if (runs === 0 || max === 0) return 0;
  return Math.max(FIELD_COLS, Math.round(Math.sqrt(runs / max) * capacity));
}

/**
 * Bottom-up dots for one time bucket. Successful runs fill from the bottom in agent-colored bands sized by
 * each agent's share; failures sit on top. A stable sprinkle of gaps (seeded by bucket) keeps the field organic.
 */
export function columnDots(bucket: FieldBucket, max: number, seed: number): readonly FieldDot[] {
  const capacity = FIELD_ROWS * FIELD_COLS;
  const lit = litDots(bucket.runs, max, capacity);
  const failed =
    bucket.failed === 0 ? 0 : Math.min(lit, Math.max(FIELD_COLS, Math.round((bucket.failed / bucket.runs) * lit)));
  const ok = lit - failed;
  const agents = bucket.agents.length ? bucket.agents : [""];
  return Array.from({ length: capacity }, (_, index): FieldDot => {
    if (index >= lit) return { kind: "grid" };
    const gap = index >= FIELD_COLS && hash(seed + 1, index) % 100 < 12;
    if (gap) return { kind: "grid" };
    if (index >= ok) return { kind: "failed" };
    return { kind: "run", color: agentDotColor(agents[Math.floor((index / ok) * agents.length)]) };
  });
}

/** Top edge of a bucket's lit dots as a 0..1 fraction of the field height, for anchoring annotation leaders. */
export function columnTop(runs: number, max: number): number {
  return Math.ceil(litDots(runs, max) / FIELD_COLS) / FIELD_ROWS;
}

export function agoLabel(thenMs: number, nowMs: number): string {
  const seconds = Math.max(0, Math.floor((nowMs - thenMs) / 1000));
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}
