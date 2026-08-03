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
