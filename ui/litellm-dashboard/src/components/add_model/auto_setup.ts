import { isAutoRouterDeployment, type AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";
import { resolveAvailableModel, type AutoRouterPreset, type ModelAvailability } from "@/lib/autorouter_presets";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const TIER_NAMES = ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"] as const;
type TierName = (typeof TIER_NAMES)[number];
export type PreferredTierModels = Record<TierName, string[]>;

const CURRENT_TIER_MODELS: PreferredTierModels = {
  SIMPLE: ["gpt-5.6-luna", "claude-haiku-4-5", "gemini-3.5-flash-lite", "deepseek-v4-flash"],
  MEDIUM: ["gpt-5.6-terra", "claude-sonnet-5", "gemini-3.8-flash", "deepseek-v4-flash"],
  COMPLEX: ["gpt-6-astra", "gpt-5.6-sol", "claude-opus-5", "gemini-3.1-pro-preview", "deepseek-v4-pro", "grok-4.6"],
  REASONING: ["gpt-6-astra", "gpt-5.6-sol", "claude-opus-5", "gemini-3.1-pro-preview", "deepseek-v4-pro", "grok-4.6"],
};

const MAX_REASONING_EFFORT: Record<string, string> = {
  "gpt-6-astra": "max",
  "gpt-5.6-sol": "max",
  "claude-opus-5": "max",
  "gemini-3.1-pro-preview": "high",
  "deepseek-v4-pro": "high",
  "grok-4.6": "xhigh",
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
          [
            ...CURRENT_TIER_MODELS[tier],
            ...presets.flatMap((preset) => preset.complexity_router_config.tiers[tier]),
          ].flatMap((model) => {
            const resolved = resolveAvailableModel(model, availability);
            return resolved ? [resolved] : [];
          }),
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
  preferredByTier: PreferredTierModels,
  availability?: ModelAvailability,
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
  const selected = selectPreferredTierModels(preferredByTier, usableNames);
  if (selected === null) return null;

  const reasoningEffort =
    availability &&
    Object.entries(MAX_REASONING_EFFORT).find(
      ([model]) => resolveAvailableModel(model, availability) === selected[3],
    )?.[1];

  return {
    tiers: {
      SIMPLE: [selected[0]],
      MEDIUM: [selected[1]],
      COMPLEX: [selected[2]],
      REASONING: [selected[3]],
    },
    classifier_type: "heuristic_v2",
    ...(reasoningEffort && {
      tier_model_params: { REASONING: { [selected[3]]: { reasoning_effort: reasoningEffort } } },
    }),
  };
};
