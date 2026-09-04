import { isAutoRouterDeployment, type AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";
import { resolveAvailableModel, type AutoRouterPreset, type ModelAvailability } from "@/lib/autorouter_presets";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

type ModelCost = {
  input_cost_per_token?: number | null;
  output_cost_per_token?: number | null;
};

export type ModelCostMap = Record<string, ModelCost>;

const TIER_NAMES = ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"] as const;
type TierName = (typeof TIER_NAMES)[number];
export type PreferredTierModels = Record<TierName, string[]>;

const price = (cost: ModelCost | null | undefined): number | undefined => {
  const input = cost?.input_cost_per_token;
  const output = cost?.output_cost_per_token;
  if (typeof input !== "number" && typeof output !== "number") return undefined;
  return (input ?? 0) + (output ?? 0);
};

const deploymentPrice = (deployment: AutoRouterDeployment, costMap: ModelCostMap): number | undefined => {
  const configured = price(deployment.litellm_params) ?? price(deployment.model_info);
  if (configured !== undefined) return configured;

  const references = [
    deployment.litellm_params?.model,
    deployment.litellm_params?.base_model,
    deployment.model_info?.base_model,
  ];
  for (const reference of references) {
    if (reference && costMap[reference]) return price(costMap[reference]);
  }
  return undefined;
};

const groupPrice = (
  modelGroup: string,
  deployments: AutoRouterDeployment[],
  costMap: ModelCostMap,
): number | undefined => {
  const groupDeployments = deployments.filter(
    (deployment) =>
      deployment.model_name === modelGroup && !deployment.litellm_params?.model?.startsWith("auto_router/"),
  );
  if (groupDeployments.length === 0) return price(costMap[modelGroup]);
  const prices = groupDeployments.map((deployment) => deploymentPrice(deployment, costMap));
  if (prices.some((value) => value === undefined)) return undefined;
  const knownPrices = prices.filter((value): value is number => value !== undefined);
  return Math.max(...knownPrices);
};

const selectTierModels = (ranked: string[]): [string, string, string, string] => {
  if (ranked.length === 1) return [ranked[0], ranked[0], ranked[0], ranked[0]];
  if (ranked.length === 2) return [ranked[0], ranked[0], ranked[1], ranked[1]];
  if (ranked.length === 3) return [ranked[0], ranked[1], ranked[2], ranked[2]];

  const last = ranked.length - 1;
  return [ranked[0], ranked[Math.floor(last / 3)], ranked[Math.floor((2 * last) / 3)], ranked[last]];
};

export const buildPreferredTierModels = (
  presets: AutoRouterPreset[],
  availability: ModelAvailability,
): PreferredTierModels =>
  Object.fromEntries(
    TIER_NAMES.map((tier) => [
      tier,
      Array.from(
        new Set(
          presets.flatMap((preset) =>
            preset.complexity_router_config.tiers[tier].flatMap((model) => {
              const resolved = resolveAvailableModel(model, availability);
              return resolved ? [resolved] : [];
            }),
          ),
        ),
      ),
    ]),
  ) as PreferredTierModels;

const selectPreferredTierModels = (
  preferredByTier: PreferredTierModels,
  usableNames: ReadonlySet<string>,
): [string, string, string, string] | null => {
  const preferred = TIER_NAMES.map((tier) => preferredByTier[tier].find((name) => usableNames.has(name)));
  const candidates = preferred.flatMap((model, tier) => (model ? [{ model, tier }] : []));
  if (candidates.length === 0) return null;

  const nearest = (tier: number): string =>
    [...candidates].sort(
      (left, right) => Math.abs(left.tier - tier) - Math.abs(right.tier - tier) || left.tier - right.tier,
    )[0].model;
  return preferred.map((model, tier) => model ?? nearest(tier)) as [string, string, string, string];
};

export const buildAutomaticRouterConfig = (
  models: ModelGroup[],
  deployments: AutoRouterDeployment[],
  costMap: ModelCostMap,
  preferredByTier?: PreferredTierModels,
): ComplexityRouterConfigValue | null => {
  const autoRouterNames: ReadonlySet<string> = new Set(
    deployments
      .filter(isAutoRouterDeployment)
      .flatMap((deployment) => (deployment.model_name ? [deployment.model_name] : [])),
  );
  const names = Array.from(
    new Set(
      models
        .filter((model) => model.mode === undefined || model.mode === "chat")
        .map((model) => model.model_group)
        .filter((name) => name && !name.startsWith("auto_router/") && !autoRouterNames.has(name)),
    ),
  );
  if (names.length === 0) return null;
  const usableNames: ReadonlySet<string> = new Set(names);

  const ranked = names
    .map((name) => ({ name, price: groupPrice(name, deployments, costMap) }))
    .sort((left, right) => {
      if (left.price === undefined && right.price !== undefined) return 1;
      if (left.price !== undefined && right.price === undefined) return -1;
      if (left.price !== undefined && right.price !== undefined && left.price !== right.price) {
        return left.price - right.price;
      }
      return left.name.localeCompare(right.name);
    })
    .map(({ name }) => name);

  const selected = preferredByTier
    ? selectPreferredTierModels(preferredByTier, usableNames) ?? selectTierModels(ranked)
    : selectTierModels(ranked);

  return {
    tiers: {
      SIMPLE: [selected[0]],
      MEDIUM: [selected[1]],
      COMPLEX: [selected[2]],
      REASONING: [selected[3]],
    },
    classifier_type: "heuristic_v2",
  };
};
