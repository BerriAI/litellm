import type { ComplexityTier } from "./KeywordTierRules";
import { TIER_ORDER } from "./tier_rows";

export type TierModelParams = Record<string, unknown>;

export type TierModelParamsByTier = Record<string, Record<string, TierModelParams>>;

/**
 * Fallback offered only when a proxy does not report supported_reasoning_efforts per model group.
 * max is deliberately absent: it is opt-in per model in the model map, so a capability-blind list
 * must not offer it; a proxy that reports capabilities supplies max itself where supported.
 */
export const REASONING_EFFORT_OPTIONS = ["none", "minimal", "low", "medium", "high", "xhigh"] as const;

/**
 * Open on purpose: the valid set is per model group at runtime (supported_reasoning_efforts), and
 * hand-authored configs can carry any level, so the known literals only add autocompletion.
 */
export type ReasoningEffort = (typeof REASONING_EFFORT_OPTIONS)[number] | (string & {});

const asRecord = (raw: unknown): Record<string, unknown> | undefined =>
  typeof raw === "object" && raw !== null && !Array.isArray(raw) ? (raw as Record<string, unknown>) : undefined;

const asTierEntryObject = (entry: unknown): { model_name: string; litellm_params: TierModelParams } | undefined => {
  const record = asRecord(entry);
  if (record === undefined || typeof record.model_name !== "string" || !record.model_name) return undefined;
  return { model_name: record.model_name, litellm_params: asRecord(record.litellm_params) ?? {} };
};

/**
 * A complexity tier maps to `str | object | list[str | object]` on the backend
 * (litellm/router_strategy/complexity_router/config.py: string/object = pin; list = random pick;
 * an object is `{model_name, litellm_params}`), and the router widens a bare value to a list.
 *
 * Every UI reader of a STORED complexity_router_config must widen the same way, so this is the
 * single owner of that rule. Readers of in-memory ComplexityTiers state are already string[]
 * and do not need it.
 */
export const normalizeTierModels = (value: unknown): string[] => {
  const entries = Array.isArray(value) ? value : [value];
  return entries.flatMap((entry) => {
    if (typeof entry === "string" && entry) return [entry];
    const parsed = asTierEntryObject(entry);
    return parsed ? [parsed.model_name] : [];
  });
};

const tierEntriesWithParams = (value: unknown): [string, TierModelParams][] =>
  (Array.isArray(value) ? value : [value])
    .map(asTierEntryObject)
    .filter((entry): entry is { model_name: string; litellm_params: TierModelParams } => entry !== undefined)
    .filter((entry) => Object.keys(entry.litellm_params).length > 0)
    .map((entry) => [entry.model_name, entry.litellm_params]);

/**
 * Params can be stored two ways: inline object entries in `tiers`, or the sibling
 * `tier_model_configs` key. The backend merges them with `tier_model_configs` winning per
 * (tier, model) (config.py `_normalize_tier_model_configs`), so hydration must too.
 */
export const hydrateTierModelParams = (
  storedTiers: unknown,
  storedTierModelConfigs: unknown,
): TierModelParamsByTier | undefined => {
  const fromInline = Object.entries(asRecord(storedTiers) ?? {}).map(
    ([tier, value]) => [tier, tierEntriesWithParams(value)] as const,
  );
  const fromSibling = Object.entries(asRecord(storedTierModelConfigs) ?? {}).map(
    ([tier, value]) => [tier, tierEntriesWithParams(value)] as const,
  );
  const merged = [...fromInline, ...fromSibling].reduce<TierModelParamsByTier>(
    (byTier, [tier, entries]) =>
      entries.length === 0 ? byTier : { ...byTier, [tier]: { ...byTier[tier], ...Object.fromEntries(entries) } },
    {},
  );
  return Object.keys(merged).length > 0 ? merged : undefined;
};

/**
 * Undefined when empty rather than `{}`, so an untouched router keeps the key out of its payload;
 * tiers this editor does not render pass through rather than being dropped now the key is managed.
 */
export const serializeTierModelConfigs = (
  tiers: Record<string, string[]>,
  tierModelParams: TierModelParamsByTier | undefined,
): Record<string, { model_name: string; litellm_params: TierModelParams }[]> | undefined => {
  if (tierModelParams === undefined) return undefined;
  const serialized = Object.entries(tierModelParams)
    .map(([tier, byModel]) => {
      const selected = tier in tiers ? new Set(tiers[tier]) : undefined;
      const entries = Object.entries(byModel)
        .filter(([model, params]) => (selected === undefined || selected.has(model)) && Object.keys(params).length > 0)
        .map(([model_name, litellm_params]) => ({ model_name, litellm_params }));
      return [tier, entries] as const;
    })
    .filter(([, entries]) => entries.length > 0);
  return serialized.length > 0 ? Object.fromEntries(serialized) : undefined;
};

export const setTierModelReasoningEffort = (
  current: TierModelParamsByTier | undefined,
  tier: string,
  model: string,
  effort: ReasoningEffort | undefined,
): TierModelParamsByTier | undefined => {
  const { reasoning_effort: _dropped, ...rest } = current?.[tier]?.[model] ?? {};
  const params = effort === undefined ? rest : { ...rest, reasoning_effort: effort };
  const byModel = Object.fromEntries(
    Object.entries({ ...current?.[tier], [model]: params }).filter(([, value]) => Object.keys(value).length > 0),
  );
  const next = Object.fromEntries(
    Object.entries({ ...current, [tier]: byModel }).filter(([, value]) => Object.keys(value).length > 0),
  );
  return Object.keys(next).length > 0 ? next : undefined;
};

export const pruneTierModelParams = (
  current: TierModelParamsByTier | undefined,
  tier: string,
  selectedModels: string[],
): TierModelParamsByTier | undefined => {
  if (current?.[tier] === undefined) return current;
  const byModel = Object.fromEntries(Object.entries(current[tier]).filter(([model]) => selectedModels.includes(model)));
  const next = Object.fromEntries(
    Object.entries({ ...current, [tier]: byModel }).filter(([, value]) => Object.keys(value).length > 0),
  );
  return Object.keys(next).length > 0 ? next : undefined;
};

export const DEFAULT_TIER_LABELS: Record<ComplexityTier, string> = {
  SIMPLE: "Simple",
  MEDIUM: "Medium",
  COMPLEX: "Complex",
  REASONING: "Reasoning",
};

export const tierOptions = (
  tierLabels: Partial<Record<ComplexityTier, string>> | undefined,
): { value: ComplexityTier; label: string }[] =>
  TIER_ORDER.map((tier) => ({ value: tier, label: tierLabels?.[tier]?.trim() || DEFAULT_TIER_LABELS[tier] }));
