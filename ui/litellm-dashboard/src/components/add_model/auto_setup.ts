import { isAutoRouterDeployment, type AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";
import { resolveAvailableModel, type AutoRouterPreset, type ModelAvailability } from "@/lib/autorouter_presets";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const TIER_NAMES = ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"] as const;
type TierName = (typeof TIER_NAMES)[number];
export type PreferredTierModels = Record<TierName, string[]>;

const ADDITIONAL_TIER_MODELS: PreferredTierModels = {
  SIMPLE: ["gpt-4o-mini", "gpt-5-mini", "gemini-2.5-flash", "deepseek-chat"],
  MEDIUM: ["gpt-5-mini", "gpt-4o", "claude-sonnet-4-5", "gemini-2.5-flash", "deepseek-chat"],
  COMPLEX: ["gpt-5", "gpt-4o", "claude-sonnet-4-6", "gemini-2.5-pro", "grok-4"],
  REASONING: ["o3", "deepseek-reasoner", "claude-opus-4-6", "gemini-2.5-pro", "gpt-5"],
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
            ...presets.flatMap((preset) => preset.complexity_router_config.tiers[tier]),
            ...ADDITIONAL_TIER_MODELS[tier],
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
