import type { CustomDimensionRow } from "./custom_dimensions";

export type TierBoundaries = Record<string, number>;

export type TokenThresholds = Record<string, number>;

export type DimensionWeights = Record<string, number>;

/**
 * Display names for the scorer's dimensions. Only the wording lives here; the dimension set and its
 * shipped weights come from the proxy (GET /public/complexity_router/scorer_defaults), so a dimension
 * added backend-side still renders, under its raw key until it is given a label here.
 */
export const DIMENSION_LABELS: Record<string, string> = {
  codePresence: "Code presence",
  reasoningMarkers: "Reasoning markers",
  technicalTerms: "Technical terms",
  tokenCount: "Token count",
  simpleIndicators: "Simple indicators",
  multiStepPatterns: "Multi-step patterns",
  questionComplexity: "Question complexity",
};

export const dimensionLabel = (key: string): string => DIMENSION_LABELS[key] ?? key;

const asRecord = (raw: unknown): Record<string, unknown> | undefined =>
  typeof raw === "object" && raw !== null && !Array.isArray(raw) ? (raw as Record<string, unknown>) : undefined;

/**
 * Absent means the router is tracking the shipped defaults, so it must hydrate to undefined rather than to
 * a copy of them: hydrating defaults would make an untouched save write them out and pin the router to
 * whatever they were the day the modal was opened. A stored weight map replaces the defaults;
 * missing dimension names score zero.
 */
const hydrateNumericMap = (raw: unknown): Record<string, number> | undefined => {
  const stored = asRecord(raw);
  if (stored === undefined) return undefined;
  return Object.fromEntries(
    Object.entries(stored).filter(([, value]) => typeof value === "number" && Number.isFinite(value)),
  ) as Record<string, number>;
};

export const hydrateTierBoundaries = (raw: unknown): TierBoundaries | undefined => hydrateNumericMap(raw);

export const hydrateTokenThresholds = (raw: unknown): TokenThresholds | undefined => hydrateNumericMap(raw);

export const hydrateDimensionWeights = (raw: unknown): DimensionWeights | undefined => hydrateNumericMap(raw);

/**
 * The scalar counterpart of hydrateNumericMap: absent hydrates to undefined so an untouched save keeps the
 * floor tracking tier_boundaries.simple_medium, while a stored 0 hydrates to 0, which is a real floor.
 */
export const hydrateReasoningOverrideMinScore = (raw: unknown): number | undefined =>
  typeof raw === "number" && Number.isFinite(raw) ? raw : undefined;

export const weightTotal = (weights: DimensionWeights): number =>
  Math.round(Object.values(weights).reduce((total, weight) => total + weight, 0) * 100) / 100;

export const effectiveDimensionWeights = (
  defaults: DimensionWeights,
  stored: DimensionWeights | undefined,
): DimensionWeights =>
  Object.fromEntries(
    Object.keys(defaults).map((name) => [name, stored === undefined ? defaults[name] : stored[name] ?? 0]),
  );

type WeightTarget = { kind: "builtin" | "custom"; id: string };
const weightValid = ({ kind, weight }: { kind: WeightTarget["kind"]; weight: number }): boolean => {
  const inRange = Number.isFinite(weight) && weight >= 0 && weight <= 1;
  return inRange && (kind === "builtin" || weight > 0);
};
export type WeightEdit =
  | { type: "set"; target: WeightTarget; weight: number }
  | { type: "add"; row: CustomDimensionRow }
  | { type: "remove"; id: string };
type WeightResult =
  | { ok: true; dimension_weights: DimensionWeights; custom_dimensions: CustomDimensionRow[] | undefined }
  | { ok: false; error: string };

export const rebalanceDimensionWeights = (
  defaults: DimensionWeights | undefined,
  stored: DimensionWeights | undefined,
  custom: CustomDimensionRow[] | undefined,
  edit: WeightEdit,
): WeightResult => {
  if (!defaults || !Object.keys(defaults).length)
    return { ok: false, error: "Load the shipped defaults before changing weights" };
  const builtin = effectiveDimensionWeights(defaults, stored);
  const rows =
    edit.type === "add"
      ? [...(custom ?? []), edit.row]
      : (custom ?? []).filter((row) => edit.type !== "remove" || row.id !== edit.id);
  const vector = [
    ...Object.entries(builtin).map(([id, weight]) => ({ kind: "builtin" as const, id, weight })),
    ...rows.map(({ id, weight }) => ({ kind: "custom" as const, id, weight })),
  ];
  if (!vector.every(weightValid))
    return {
      ok: false,
      error: "Existing weights must be finite and nonnegative; custom weights must be greater than 0 and at most 1",
    };
  const target = edit.type === "set" ? edit.target : undefined;
  const pinned = (entry: WeightTarget) =>
    edit.type === "add"
      ? entry.kind === "custom" && entry.id === edit.row.id
      : entry.kind === target?.kind && entry.id === target.id;
  if (edit.type === "set" && !vector.some(pinned)) return { ok: false, error: "The dimension is no longer available" };
  const requestedWeight = (): number => {
    if (edit.type === "set") return edit.weight;
    return edit.type === "add" ? edit.row.weight : 0;
  };
  const weight = requestedWeight();
  if (!weightValid({ kind: target?.kind ?? "builtin", weight }))
    return { ok: false, error: "Use a weight from 0 to 1; custom dimensions must stay greater than 0" };
  const others = vector.filter((entry) => !pinned(entry));
  const total = others.reduce((sum, entry) => sum + entry.weight, 0);
  if (!Number.isFinite(total)) return { ok: false, error: "Existing weights are too large to rebalance" };
  const remainder = 1 - weight;
  const builtinCount = others.filter((entry) => entry.kind === "builtin").length;
  const redistributed = (entry: WeightTarget & { weight: number }): number => {
    if (pinned(entry)) return weight;
    if (total > 0) return remainder * (entry.weight / total);
    return entry.kind === "builtin" ? remainder / builtinCount : 0;
  };
  const balanced = vector.map((entry) => ({ ...entry, weight: redistributed(entry) }));
  const residual = 1 - balanced.reduce((sum, entry) => sum + entry.weight, 0);
  const receiver = balanced
    .filter((entry) => entry.kind === "builtin" && !pinned(entry) && entry.weight > 0)
    .sort((a, b) => b.weight - a.weight)[0];
  const corrected = balanced.map((entry) =>
    entry === receiver ? { ...entry, weight: entry.weight + residual } : entry,
  );
  const offBudget = Math.abs(corrected.reduce((sum, entry) => sum + entry.weight, 0) - 1) > 1e-12;
  if (!corrected.every(weightValid) || offBudget)
    return { ok: false, error: "Leave a positive share for every custom dimension, or remove it first" };
  const weights = Object.fromEntries(
    corrected.filter((entry) => entry.kind === "builtin").map(({ id, weight }) => [id, weight]),
  );
  const customWeights = new Map(
    corrected.filter((entry) => entry.kind === "custom").map(({ id, weight }) => [id, weight]),
  );
  const emptyRows = edit.type === "remove" ? undefined : custom;
  return {
    ok: true,
    dimension_weights: { ...stored, ...weights },
    custom_dimensions: rows.length ? rows.map((row) => ({ ...row, weight: customWeights.get(row.id)! })) : emptyRows,
  };
};
