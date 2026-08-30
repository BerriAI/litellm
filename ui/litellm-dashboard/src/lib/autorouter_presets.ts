import {
  ComplexityRouterConfigPayload,
  hydrateTierLabels,
} from "@/components/add_model/build_complexity_router_config";
import {
  ComplexityRouterConfigValue,
  ClassifierType,
  ClassifierLLMConfig,
  DEFAULT_SESSION_AFFINITY,
  DEFAULT_DEPLOYMENT_AFFINITY,
  usesLlmClassifier,
} from "@/components/add_model/ComplexityRouterConfig";
import { KeywordTierRule } from "@/components/add_model/KeywordTierRules";
import { hydrateKeywordTierRules } from "@/components/add_model/complexity_router_keywords";
import {
  TierModelParams,
  TierModelParamsByTier,
  hydrateTierModelParams,
} from "@/components/add_model/complexity_router_tiers";
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
  config: Pick<ComplexityRouterConfigPayload, "tiers" | "classifier_llm_config" | "embedding_model" | "default_model">,
): Set<string> => {
  const { tiers, classifier_llm_config: classifier, embedding_model: embedding, default_model: pinned } = config;
  const models = [...Object.values(tiers).flat(), classifier?.model, embedding, pinned];
  // Boolean(), not != null: an empty-string placeholder (e.g. classifier_llm_config seeded before a
  // model is chosen) is never a real model reference either.
  return new Set(models.filter((model): model is string => Boolean(model)));
};

// Admins spell version numbers inconsistently ("claude-sonnet-4-5" vs "claude-sonnet-4.5"), so a
// preset's hardcoded name and a caller's registered one can refer to the same model while
// differing only in that separator. Canonicalizing on "-" (the presets' own convention) lets both
// spellings match without doing anything looser - two DIFFERENT model names never collide here,
// only the punctuation within one version number does.
export const normalizeModelName = (model: string): string => model.replace(/(\d)\.(\d)/g, "$1-$2");

export interface DeploymentModelRef {
  modelGroup: string;
  underlyingModels: readonly string[];
}

export interface ModelAvailability {
  modelGroups: Set<string>;
  underlyingIndex: Map<string, readonly string[]>;
}

const normalizeUnderlyingModel = (model: string): string | null => {
  if (model.includes("*")) return null;
  const ownName = model.slice(model.lastIndexOf("/") + 1).split("@")[0];
  const lastNamespaceSegment = normalizeModelName(ownName).split(".").at(-1) ?? "";
  const stripped = lastNamespaceSegment
    .replace(/:\d+k$/i, "")
    .replace(/\[\w+\]$/, "")
    .replace(/-v\d+(:\d+)?$/, "")
    .replace(/-20\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$/, "");
  return stripped.toLowerCase() || null;
};

// A linear glob scan rather than a RegExp: patterns are admin-controlled model_name values, and a
// backtracking regex built from one ("a*a*a*...") can freeze another admin's dashboard.
const matchesWildcard = (pattern: string, name: string): boolean => {
  const parts = pattern.split("*");
  if (parts.length === 1) return pattern === name;
  const head = parts[0];
  const tail = parts[parts.length - 1];
  if (!name.startsWith(head) || !name.endsWith(tail)) return false;
  if (name.length < head.length + tail.length) return false;
  const scanEnd = name.length - tail.length;
  const scanResult = parts.slice(1, -1).reduce((searchFrom: number, part: string) => {
    if (searchFrom < 0) return -1;
    const found = name.indexOf(part, searchFrom);
    return found === -1 || found + part.length > scanEnd ? -1 : found + part.length;
  }, head.length);
  return scanResult >= 0;
};

export const buildModelAvailability = (
  modelGroups: Iterable<string>,
  deployments: readonly DeploymentModelRef[],
): ModelAvailability => {
  const groups = new Set(modelGroups);
  const literalEntries = deployments
    .filter((deployment) => groups.has(deployment.modelGroup))
    .flatMap((deployment) =>
      deployment.underlyingModels
        .map(normalizeUnderlyingModel)
        .filter((key): key is string => key !== null)
        .map((key) => ({ key, modelGroup: deployment.modelGroup })),
    );
  // Mirrors get_known_models_from_wildcard: a bare "*" model_name expands via its underlying
  // wildcard (or not at all), and a wildcard without a "/" expands to nothing.
  const wildcardPatterns = Array.from(
    new Set(
      deployments
        .flatMap((deployment) =>
          deployment.modelGroup === "*" ? deployment.underlyingModels : [deployment.modelGroup],
        )
        .filter((pattern) => pattern !== "*" && pattern.includes("*") && pattern.includes("/")),
    ),
  );
  const wildcardEntries = Array.from(groups)
    .filter((group) => !group.includes("*") && wildcardPatterns.some((pattern) => matchesWildcard(pattern, group)))
    .map((group) => ({ key: normalizeUnderlyingModel(group), modelGroup: group }))
    .filter((entry): entry is { key: string; modelGroup: string } => entry.key !== null);
  const entries = [...literalEntries, ...wildcardEntries];
  const grouped = new Map<string, Set<string>>();
  for (const entry of entries) {
    const groupsForKey = grouped.get(entry.key) ?? new Set<string>();
    groupsForKey.add(entry.modelGroup);
    grouped.set(entry.key, groupsForKey);
  }
  const underlyingIndex = new Map(
    Array.from(grouped, ([key, groupsForKey]) => [key, Array.from(groupsForKey).sort()] as const),
  );
  return { modelGroups: groups, underlyingIndex };
};

export const deploymentRefsFromModelInfo = (
  rows: readonly {
    model_name?: string | null;
    litellm_params?: { model?: string | null; base_model?: string | null } | null;
    model_info?: { base_model?: string | null } | null;
  }[],
): DeploymentModelRef[] =>
  rows.flatMap((row) => {
    const underlyingModels = [
      row.litellm_params?.model,
      row.litellm_params?.base_model,
      row.model_info?.base_model,
    ].filter((model): model is string => Boolean(model));
    return row.model_name && underlyingModels.length > 0 ? [{ modelGroup: row.model_name, underlyingModels }] : [];
  });

const resolveAvailableModel = (requiredModel: string, availability: ModelAvailability): string | undefined => {
  const { modelGroups, underlyingIndex } = availability;
  if (modelGroups.has(requiredModel)) return requiredModel;
  const normalized = normalizeModelName(requiredModel);
  const groupMatch = Array.from(modelGroups).find((available) => normalizeModelName(available) === normalized);
  if (groupMatch !== undefined) return groupMatch;
  const key = normalizeUnderlyingModel(requiredModel);
  return key === null ? undefined : underlyingIndex.get(key)?.[0];
};

export const getMissingModels = (
  config: Parameters<typeof getRequiredModels>[0],
  availability: ModelAvailability,
): string[] =>
  [...getRequiredModels(config)].filter((model) => resolveAvailableModel(model, availability) === undefined).sort();

export const getRequiredModelsInPreset = (preset: AutoRouterPreset): Set<string> =>
  getRequiredModels(preset.complexity_router_config);

export const getMissingModelsInPreset = (preset: AutoRouterPreset, availability: ModelAvailability): string[] =>
  getMissingModels(preset.complexity_router_config, availability);

// Checks the config actually being built (whether it arrived via a preset prefill or was typed by
// hand - the two are indistinguishable once the caller has started editing), not a preset's
// original bundled model list. Only counts classifier_llm_config/embedding_model as referenced
// when buildComplexityRouterConfig would actually emit them (usesLlmClassifier(classifierType),
// semanticMatchingEnabled) - otherwise a dormant selection left over from a toggle no longer in
// effect would block submit for a model that was never going to be submitted.
export const getReferencedModelsError = (
  params: {
    tiers: ComplexityRouterConfigPayload["tiers"];
    classifierType: ClassifierType;
    classifierLlmConfig: ClassifierLLMConfig | undefined;
    semanticMatchingEnabled: boolean;
    embeddingModel: string | undefined;
    defaultModel?: string;
  },
  availability: ModelAvailability,
): string | null => {
  const missing = getMissingModels(
    {
      tiers: params.tiers,
      default_model: params.defaultModel,
      classifier_llm_config: usesLlmClassifier(params.classifierType) ? params.classifierLlmConfig : undefined,
      embedding_model: params.semanticMatchingEnabled ? params.embeddingModel : undefined,
    },
    availability,
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
// `availability` is required, not optional: every model reference gets rewritten to the
// caller's actual registered spelling (resolveAvailableModel), which may differ from the preset's
// literal string by version-separator punctuation alone. Called only after presetAvailability has
// already confirmed every required model resolves, so falling back to the preset's own string
// when a model somehow doesn't resolve is unreachable in practice, not a silent-failure path.
export const buildPresetPrefill = (
  config: ComplexityRouterConfigPayload,
  availability: ModelAvailability,
): PresetPrefill => {
  const resolve = (model: string): string => resolveAvailableModel(model, availability) ?? model;
  const resolveTier = (models: string[]): string[] => models.map(resolve);
  // Params key on the model name the preset spells while every tier entry is rewritten to the
  // caller's registered spelling, so the keys have to be rewritten the same way. Otherwise
  // serializeTierModelConfigs drops them for naming a model the tier no longer holds.
  //
  // Two spellings in one tier can resolve to the same registered model, and one model holds one
  // param set here and in the payload, so a collision has to collapse. Merge rather than replace:
  // params only one spelling set still survive, and a key both set resolves last-wins, matching
  // how hydrateTierModelParams already collapses two entries spelled identically.
  const resolveParamKeys = (params: TierModelParamsByTier | undefined): TierModelParamsByTier | undefined =>
    params &&
    Object.fromEntries(
      Object.entries(params).map(([tier, byModel]) => [
        tier,
        Object.entries(byModel).reduce<Record<string, TierModelParams>>((byResolved, [model, litellmParams]) => {
          const resolved = resolve(model);
          return { ...byResolved, [resolved]: { ...byResolved[resolved], ...litellmParams } };
        }, {}),
      ]),
    );

  return {
    complexityRouterConfig: {
      tiers: {
        SIMPLE: resolveTier(config.tiers.SIMPLE),
        MEDIUM: resolveTier(config.tiers.MEDIUM),
        COMPLEX: resolveTier(config.tiers.COMPLEX),
        REASONING: resolveTier(config.tiers.REASONING),
      },
      tier_model_params: resolveParamKeys(hydrateTierModelParams(config.tiers, config.tier_model_configs)),
      tier_labels: hydrateTierLabels(config.tier_labels),
      classifier_type: config.classifier_type,
      classifier_llm_config: config.classifier_llm_config && {
        ...config.classifier_llm_config,
        model: resolve(config.classifier_llm_config.model),
      },
      classifier_context_window_size: config.classifier_context_window_size,
      classifier_context_budget_chars: config.classifier_context_budget_chars,
      classifier_context_per_turn_chars: config.classifier_context_per_turn_chars,
      classifier_context_include_assistant_turns: config.classifier_context_include_assistant_turns,
      session_affinity: config.session_affinity ?? DEFAULT_SESSION_AFFINITY,
      deployment_affinity: config.deployment_affinity ?? DEFAULT_DEPLOYMENT_AFFINITY,
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
