export interface ModelThrottlingThreshold {
  spend: number;
  models: string[];
}

export interface ModelThrottling {
  thresholds: ModelThrottlingThreshold[];
}

export function modelThrottlingError(value: ModelThrottling | null | undefined, maxBudget: unknown): string | null {
  if (!value) return null;
  const limit = Number(maxBudget);
  if (!Number.isFinite(limit) || limit <= 0) return "Set a final budget before enabling model throttling";
  if (value.thresholds.length === 0) return "Add a throttling threshold";
  if (value.thresholds.some(({ spend }) => !Number.isFinite(spend) || spend <= 0 || spend >= limit)) {
    return "Each threshold must be above $0 and below the final budget";
  }
  if (value.thresholds.some(({ models }) => models.length === 0)) return "Select models to pause at each threshold";
  const amounts = value.thresholds.map(({ spend }) => spend);
  if (new Set(amounts).size !== amounts.length) return "Combine models at the same spend into one threshold";
  const models = value.thresholds.flatMap((threshold) => threshold.models);
  if (new Set(models).size !== models.length) return "Select each model at only one threshold";
  return null;
}

export function modelsWithinAccess(
  available: string[],
  allowed: string[] | null | undefined,
  groups: Readonly<Record<string, readonly string[]>> = {},
  grantedModels: readonly string[] = [],
): string[] {
  const expand = (models: readonly string[]) => models.flatMap((model) => groups[model] ?? [model]);
  const concrete = [...new Set(expand(available))].filter(
    (model) => !model.includes("*") && !["all-proxy-models", "all-team-models", "no-default-models"].includes(model),
  );
  if (
    (!allowed?.length && !grantedModels.length) ||
    allowed?.some((model) => ["*", "all-proxy-models", "all-team-models"].includes(model))
  )
    return concrete;
  const grants = [...expand(allowed ?? []), ...grantedModels];
  return concrete.filter((model) =>
    grants.some((grant) => grant === model || (grant.endsWith("/*") && model.startsWith(grant.slice(0, -1)))),
  );
}

export function modelThrottlingUpdate(value: ModelThrottling | null, stored: ModelThrottling | null | undefined) {
  return JSON.stringify(value) === JSON.stringify(stored ?? null) ? {} : { model_throttling: value };
}
