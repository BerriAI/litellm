import type { ComplexityTiers } from "./ComplexityRouterConfig";

/**
 * A complexity tier maps to `str | list[str]` on the backend
 * (litellm/router_strategy/complexity_router/config.py: "string = pin; list = random pick"),
 * and the router widens the bare string with `models if isinstance(models, list) else [models]`.
 *
 * Every UI reader of a STORED complexity_router_config must widen the same way, so this is the
 * single owner of that rule. Readers of in-memory ComplexityTiers state are already string[]
 * and do not need it.
 */
export const normalizeTierModels = (value: unknown): string[] => {
  if (Array.isArray(value)) return value.filter((model): model is string => typeof model === "string");
  if (typeof value === "string" && value) return [value];
  return [];
};

/**
 * Mirrors `init_complexity_router_deployment` (litellm/router.py): an explicit pin wins, otherwise
 * the default is `MEDIUM or SIMPLE`. Deriving past SIMPLE would name a model the backend never
 * picks, and it raises rather than falling through to COMPLEX/REASONING.
 */
export const resolveComplexityDefaultModel = (tiers: ComplexityTiers, pinned?: string): string | undefined =>
  pinned?.trim() || tiers.MEDIUM[0] || tiers.SIMPLE[0];
