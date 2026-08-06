import { ComplexityRouterConfigPayload } from "@/components/add_model/build_complexity_router_config";
import {
  ComplexityRouterConfigValue,
  ComplexityTiers,
  ClassifierType,
  ClassifierLLMConfig,
  DEFAULT_SESSION_AFFINITY,
} from "@/components/add_model/ComplexityRouterConfig";
import { KeywordTierRule } from "@/components/add_model/KeywordTierRules";
import { hydrateKeywordTierRules } from "@/components/add_model/complexity_router_keywords";
import { DEFAULT_ESCALATION_KEYWORDS } from "@/components/add_model/EscalationKeywords";
import { DEFAULT_MATCH_THRESHOLD } from "@/components/add_model/SemanticKeywordMatching";
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
// hence this one cast). autorouter_presets.test.ts pins the parsed shape, so a JSON typo fails CI.
const RAW = presetsRaw as Record<string, Omit<AutoRouterPreset, "key">>;

const PRESETS: AutoRouterPreset[] = Object.entries(RAW).map(([key, preset]) => ({ key, ...preset }));

export const getAllPresets = (): AutoRouterPreset[] => PRESETS;

export const getPresetByKey = (key: string): AutoRouterPreset | undefined => PRESETS.find((p) => p.key === key);

// Generalized over ComplexityRouterConfigPayload so the same accessors check either a preset's own
// bundled config or a caller's actually-built config - the two need to agree, since a preset only
// prefills once and the config is edited freely after (see AddAutoRouterTab.submitBlockedReason).
export const getRequiredModels = (
  config: Pick<ComplexityRouterConfigPayload, "tiers" | "classifier_llm_config" | "embedding_model">,
): Set<string> => {
  const { tiers, classifier_llm_config: classifier, embedding_model: embedding } = config;
  const models = [...tiers.SIMPLE, ...tiers.MEDIUM, ...tiers.COMPLEX, ...tiers.REASONING, classifier?.model, embedding];
  // Boolean(), not != null: an empty-string placeholder (e.g. classifier_llm_config seeded before a
  // model is chosen) is never a real model reference either.
  return new Set(models.filter((model): model is string => Boolean(model)));
};

// Admins spell version numbers inconsistently ("claude-sonnet-4-5" vs "claude-sonnet-4.5"), so a
// preset's hardcoded name and a caller's registered one can refer to the same model while
// differing only in that separator. Canonicalizing on "-" (the presets' own convention) lets both
// spellings match without doing anything looser - two DIFFERENT model names never collide here,
// only the punctuation within one version number does.
const normalizeModelName = (model: string): string => model.replace(/(\d)\.(\d)/g, "$1-$2");

// The caller's actual registered spelling for a required model, under either separator
// convention, or undefined if truly absent. Preset prefill must write THIS spelling, not the
// preset's literal string - otherwise a caller whose proxy only has the dotted form ends up with
// a tier pointing at a model name that was never registered.
const resolveAvailableModel = (requiredModel: string, availableModels: Set<string>): string | undefined => {
  if (availableModels.has(requiredModel)) return requiredModel;
  const normalized = normalizeModelName(requiredModel);
  return Array.from(availableModels).find((available) => normalizeModelName(available) === normalized);
};

export const getMissingModels = (
  config: Pick<ComplexityRouterConfigPayload, "tiers" | "classifier_llm_config" | "embedding_model">,
  availableModels: Set<string>,
): string[] =>
  [...getRequiredModels(config)].filter((model) => resolveAvailableModel(model, availableModels) === undefined).sort();

export const getRequiredModelsInPreset = (preset: AutoRouterPreset): Set<string> =>
  getRequiredModels(preset.complexity_router_config);

export const getMissingModelsInPreset = (preset: AutoRouterPreset, availableModels: Set<string>): string[] =>
  getMissingModels(preset.complexity_router_config, availableModels);

// Checks the config actually being built (whether it arrived via a preset prefill or was typed by
// hand - the two are indistinguishable once the caller has started editing), not a preset's
// original bundled model list. Only counts classifier_llm_config/embedding_model as referenced
// when buildComplexityRouterConfig would actually emit them (classifierType === "llm",
// semanticMatchingEnabled) - otherwise a dormant selection left over from a toggle no longer in
// effect would block submit for a model that was never going to be submitted.
export const getReferencedModelsError = (
  params: {
    tiers: ComplexityTiers;
    classifierType: ClassifierType;
    classifierLlmConfig: ClassifierLLMConfig | undefined;
    semanticMatchingEnabled: boolean;
    embeddingModel: string | undefined;
  },
  availableModels: Set<string>,
): string | null => {
  const missing = getMissingModels(
    {
      tiers: params.tiers,
      classifier_llm_config: params.classifierType === "llm" ? params.classifierLlmConfig : undefined,
      embedding_model: params.semanticMatchingEnabled ? params.embeddingModel : undefined,
    },
    availableModels,
  );
  return missing.length > 0 ? `Model(s) no longer available: ${missing.join(", ")}` : null;
};

// Every piece of AddAutoRouterTab's config state that a preset (or a reset to Custom) prefills in
// one shot, so handlePresetChange has exactly one thing to apply rather than seven setters to keep
// in sync by hand.
export interface PresetPrefill {
  complexityRouterConfig: ComplexityRouterConfigValue;
  customTechnicalKeywords: string[];
  keywordTierRules: KeywordTierRule[];
  semanticMatchingEnabled: boolean;
  embeddingModel: string | undefined;
  matchThreshold: number;
  escalationKeywords: string[];
}

export const buildEmptyPrefill = (): PresetPrefill => ({
  complexityRouterConfig: {
    tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
    classifier_type: "heuristic",
  },
  customTechnicalKeywords: [],
  keywordTierRules: [],
  semanticMatchingEnabled: false,
  embeddingModel: undefined,
  matchThreshold: DEFAULT_MATCH_THRESHOLD,
  escalationKeywords: DEFAULT_ESCALATION_KEYWORDS,
});

// `??`, never `||`: a preset's match_threshold: 0 or escalation_keywords: [] is a deliberate,
// falsy value that must survive the prefill, not get silently replaced by the default.
//
// `availableModels` is required, not optional: every model reference gets rewritten to the
// caller's actual registered spelling (resolveAvailableModel), which may differ from the preset's
// literal string by version-separator punctuation alone. Called only after presetAvailability has
// already confirmed every required model resolves, so falling back to the preset's own string
// when a model somehow doesn't resolve is unreachable in practice, not a silent-failure path.
export const buildPresetPrefill = (
  config: ComplexityRouterConfigPayload,
  availableModels: Set<string>,
): PresetPrefill => {
  const resolve = (model: string): string => resolveAvailableModel(model, availableModels) ?? model;
  const resolveTier = (models: string[]): string[] => models.map(resolve);

  return {
    complexityRouterConfig: {
      tiers: {
        SIMPLE: resolveTier(config.tiers.SIMPLE),
        MEDIUM: resolveTier(config.tiers.MEDIUM),
        COMPLEX: resolveTier(config.tiers.COMPLEX),
        REASONING: resolveTier(config.tiers.REASONING),
      },
      classifier_type: config.classifier_type,
      classifier_llm_config: config.classifier_llm_config && {
        ...config.classifier_llm_config,
        model: resolve(config.classifier_llm_config.model),
      },
      classifier_context_window_size: config.classifier_context_window_size,
      classifier_context_per_turn_chars: config.classifier_context_per_turn_chars,
      classifier_context_include_assistant_turns: config.classifier_context_include_assistant_turns,
      session_affinity: config.session_affinity ?? DEFAULT_SESSION_AFFINITY,
      adaptive: config.adaptive,
      adaptive_weights: config.adaptive_weights,
      tier_distance_penalty: config.tier_distance_penalty,
      adaptive_eligible: config.adaptive_eligible,
      return_raw_model_name: config.return_raw_model_name,
    },
    customTechnicalKeywords: config.custom_technical_keywords ?? [],
    keywordTierRules: hydrateKeywordTierRules(config.keyword_tier_rules ?? []),
    semanticMatchingEnabled: config.semantic_keyword_matching ?? false,
    embeddingModel: config.embedding_model && resolve(config.embedding_model),
    matchThreshold: config.match_threshold ?? DEFAULT_MATCH_THRESHOLD,
    escalationKeywords: config.escalation_keywords ?? DEFAULT_ESCALATION_KEYWORDS,
  };
};
