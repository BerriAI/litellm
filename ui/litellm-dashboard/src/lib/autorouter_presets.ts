import { ComplexityRouterConfigPayload } from "@/components/add_model/build_complexity_router_config";
import presetsRaw from "@/autorouter_presets.json";

// `key` is the stable JSON object key (e.g. "anthropic_family"); `label` is display text and
// never an identity.
export interface AutoRouterPreset {
  key: string;
  label: string;
  description: string;
  complexity_router_config: ComplexityRouterConfigPayload;
}

// The bundled JSON is a developer-authored, build-time asset, so it is trusted at the import
// boundary rather than re-validated at runtime (resolveJsonModule widens its string literals,
// hence this one cast). autorouter_presets.test.ts pins the parsed shape, so a JSON typo fails
// CI.
const RAW = presetsRaw as Record<string, Omit<AutoRouterPreset, "key">>;

const PRESETS: AutoRouterPreset[] = Object.entries(RAW).map(([key, preset]) => ({ key, ...preset }));

export const getAllPresets = (): AutoRouterPreset[] => PRESETS;

export const getPresetByKey = (key: string): AutoRouterPreset | undefined => PRESETS.find((p) => p.key === key);

// Generalized over ComplexityRouterConfigPayload (a preset's bundled config) so the same accessors
// can check either a preset's own models or a caller's actually-built config - the two need to
// agree, since a preset only prefills once and the config is edited freely after.
export const getRequiredModels = (
  config: Pick<ComplexityRouterConfigPayload, "tiers" | "classifier_llm_config" | "embedding_model">,
): Set<string> => {
  const { tiers, classifier_llm_config: classifier, embedding_model: embedding } = config;
  const models = [...tiers.SIMPLE, ...tiers.MEDIUM, ...tiers.COMPLEX, ...tiers.REASONING, classifier?.model, embedding];
  // Boolean(), not != null: an empty-string placeholder (e.g. classifier_llm_config seeded before a
  // model is chosen) is never a real model reference either.
  return new Set(models.filter((model): model is string => Boolean(model)));
};

export const getMissingModels = (
  config: Pick<ComplexityRouterConfigPayload, "tiers" | "classifier_llm_config" | "embedding_model">,
  availableModels: Set<string>,
): string[] => [...getRequiredModels(config)].filter((model) => !availableModels.has(model)).sort();

export const getRequiredModelsInPreset = (preset: AutoRouterPreset): Set<string> =>
  getRequiredModels(preset.complexity_router_config);

export const getMissingModelsInPreset = (preset: AutoRouterPreset, availableModels: Set<string>): string[] =>
  getMissingModels(preset.complexity_router_config, availableModels);
