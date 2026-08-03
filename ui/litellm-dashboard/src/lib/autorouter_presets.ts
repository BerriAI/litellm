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

export const getRequiredModelsInPreset = (preset: AutoRouterPreset): Set<string> => {
  const { tiers, classifier_llm_config: classifier, embedding_model: embedding } = preset.complexity_router_config;
  const models = [...tiers.SIMPLE, ...tiers.MEDIUM, ...tiers.COMPLEX, ...tiers.REASONING, classifier?.model, embedding];
  return new Set(models.filter((model): model is string => model != null));
};

export const getMissingModelsInPreset = (preset: AutoRouterPreset, availableModels: Set<string>): string[] =>
  [...getRequiredModelsInPreset(preset)].filter((model) => !availableModels.has(model)).sort();
