import { useLanguage } from "@/contexts/LanguageContext";
import RoutingOptions from "./RoutingOptions";
import ClassifierPrimarySettings from "./ClassifierPrimarySettings";
import { AutoRouterAllowanceNote } from "./AutoRouterAvailability";
import type { JevClassifierConfig } from "./jev_classifier_config";
import { isHeuristicChain, type ClassifierType, type LocalHeuristic } from "./classifier_types";
export { type ClassifierType, usesLlmClassifier, usesClassifierContext } from "./classifier_types";
import ForecastClassifierConfig, { ForecastSolverModels } from "./ForecastClassifierConfig";
import { isForecastClassifier, type CapabilitySettings, type FuseSettings } from "./forecast_classifier_config";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { MultiSelect } from "@/components/shared/MultiSelect";
import TierConfigIntro from "./TierConfigIntro";
import DefaultModelField from "./DefaultModelField";
import { Info, Plus, Trash2, X } from "lucide-react";

import NonReasoningTierToggle from "./NonReasoningTierToggle";
import TierRowSelect from "./TierRowSelect";
import { Card, CardContent } from "@/components/ui/card";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  type CustomTierSet,
  type TierRow,
  MAX_TIER_COUNT,
  MAX_TIER_DEFINITION_CHARS,
  MAX_TIER_NAME_CHARS,
  ALL_BUILT_IN_TIERS,
  MIN_TIER_COUNT,
  TIER_ORDER,
  activeTierName,
  activeTierRows,
  getCustomTierRowsError,
  isBuiltInTierName,
  resolveComplexityDefaultModel,
} from "./tier_rows";
import React from "react";
import { ModelGroup } from "@/components/llm_calls/fetch_models";
import ClassificationMethodConfig, { InactiveHeuristicV2Threshold } from "./ClassificationMethodConfig";
import ComplexityRouterAdvancedSections from "./ComplexityRouterAdvancedSections";
import { type TierSetAction, applyTierSetAction, setFallbackTier } from "./tier_set_actions";
import {
  ReasoningEffort,
  TierModelParamChange,
  TierModelParamsByTier,
  classifierEffortOptionsForModels,
  setTierModelParam,
  tierEffortOptionsForModels,
  tierRowLabel,
} from "./complexity_router_tiers";
import TierModelEffortRows from "./TierModelEffortRows";
import { KeywordTierRule } from "./KeywordTierRules";
import { type DimensionWeights, type TierBoundaries, type TokenThresholds } from "./heuristic_scoring_knobs";
import { type CustomDimensionRow } from "./custom_dimensions";
import { type AutoRouterCompressionState, DEFAULT_AUTO_ROUTER_COMPRESSION } from "./buildAutoRouterCompression";
import { type ReminderMarkerPair } from "./build_complexity_router_config";

export type { DimensionWeights, TierBoundaries, TokenThresholds };
export type { CustomTierSet, TierRow } from "./tier_rows";
export type { ReminderMarkerPair } from "./build_complexity_router_config";

export const DEFAULT_CLASSIFIER_TIMEOUT_MS = 3000;
export const DEFAULT_TIER_DISTANCE_PENALTY = 0.5;
export const DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE = 3;
export const DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS = 8000;
export const MIN_QUOTED_CONTEXT_TURN_CHARS = 120;
export const DEFAULT_SESSION_AFFINITY = false;
export const DEFAULT_SESSION_AFFINITY_TTL_SECONDS = 3600;
export const DEFAULT_DEPLOYMENT_AFFINITY = true;

export type ClassificationMode = "every_request" | "user_turn";

export const DEFAULT_CLASSIFICATION_MODE: ClassificationMode = "every_request";

/**
 * One operator-facing choice over the two wire fields that share the router's tier-pin machinery:
 * session affinity pins every turn, user_turn pins every turn except a new human ask.
 */
export type ClassificationFrequency = ClassificationMode | "session";

/** NON_REASONING is optional: a router that never enabled it stores no such key. */
export type ComplexityTiers = {
  SIMPLE: string[];
  MEDIUM: string[];
  COMPLEX: string[];
  REASONING: string[];
  NON_REASONING?: string[];
};

import type { ClassificationRubric } from "./classification_rubrics";
export {
  type ClassificationRubric,
  DEFAULT_CLASSIFICATION_RUBRIC,
  NEW_CLASSIFIER_CLASSIFICATION_RUBRIC,
  CLASSIFICATION_RUBRIC_DESCRIPTIONS,
  CLASSIFICATION_RUBRIC_KEYS,
} from "./classification_rubrics";

export interface ClassifierLLMConfig {
  model: string;
  timeout_ms: number;
  circuit_breaker_enabled?: boolean;
  circuit_breaker_cooldown_seconds?: number;
  reasoning_effort?: ReasoningEffort;
  classification_rubric?: ClassificationRubric;
  system_prompt?: string;
}

export type ClassifierFallback = "heuristic" | "default_model";

export const DEFAULT_CLASSIFIER_FALLBACK: ClassifierFallback = "heuristic";

export interface AdaptiveRouterWeights {
  quality: number;
  cost: number;
}

export const DEFAULT_ADAPTIVE_WEIGHTS: AdaptiveRouterWeights = { quality: 0.3, cost: 0.7 };

export type HeuristicScoringRole = "decides" | "fallback_only" | "never";

export const heuristicScoringRoleFor = (
  classifierType: ClassifierType,
  classifierFallback: ClassifierFallback | undefined,
  localHeuristic?: LocalHeuristic,
): HeuristicScoringRole => {
  if (isHeuristicChain(classifierType) && localHeuristic === "heuristic_v2") return "never";
  if (classifierType === "heuristic_v2" || classifierType === "capability" || classifierType === "llm_v2")
    return "never";
  if (classifierType === "heuristic" || classifierType === "heuristic_first" || classifierType === "hybrid")
    return "decides";
  return (classifierFallback ?? DEFAULT_CLASSIFIER_FALLBACK) === "heuristic" ? "fallback_only" : "never";
};

export const heuristicScoringRole = (value: ComplexityRouterConfigValue): HeuristicScoringRole =>
  value.custom_tier_set
    ? "never"
    : heuristicScoringRoleFor(value.classifier_type, value.classifier_fallback, value.local_heuristic);

export const usesHeuristicV2 = (value: ComplexityRouterConfigValue): boolean => {
  if (value.custom_tier_set) return false;
  return (
    value.classifier_type === "heuristic_v2" ||
    (isHeuristicChain(value.classifier_type) && value.local_heuristic === "heuristic_v2")
  );
};

export const heuristicTuningType = (value: ComplexityRouterConfigValue): LocalHeuristic | undefined => {
  if (value.custom_tier_set || value.classifier_type === "llm") return undefined;
  if (usesHeuristicV2(value)) return "heuristic_v2";
  return heuristicScoringRole(value) === "never" ? undefined : "heuristic";
};

// Derived, never written into the value, so undoing a tier edit reverts the form with nothing left behind.
export const effectiveClassifierType = (
  value: Pick<ComplexityRouterConfigValue, "custom_tier_set" | "classifier_type">,
): ClassifierType => (value.custom_tier_set && value.classifier_type !== "jev" ? "llm" : value.classifier_type);

const rowOrigin = (row: TierRow, editing: boolean): string => {
  if (!editing) return row.id;
  return isBuiltInTierName(row.name) ? "built-in" : "custom";
};

const builtInTierInfo = (rowId: string): { label: string; description: string; examples: string } | undefined => {
  const builtIn = ALL_BUILT_IN_TIERS.find((tier) => tier === rowId);
  return builtIn ? TIER_DESCRIPTIONS[builtIn] : undefined;
};

const TierSetToolbar: React.FC<{
  editing: boolean;
  isCustomSet: boolean;
  rowCount: number;
  rowsError: string | null;
  keywordRulesError: string | null | undefined;
  onEditingChange: ((editing: boolean) => void) | undefined;
  onAdd: () => void;
  onRestore: () => void;
}> = ({ editing, isCustomSet, rowCount, rowsError, keywordRulesError, onEditingChange, onAdd, onRestore }) => {
  const { t } = useLanguage();
  return (
    <>
      <div className="mt-4 flex flex-wrap items-center gap-2">
        {editing ? (
          <>
            <Button variant="outline" onClick={onAdd} disabled={rowCount >= MAX_TIER_COUNT}>
              <Plus />
              {t("Add tier")}
            </Button>
            <SimpleTooltip content={rowsError || undefined}>
              <Button variant="outline" disabled={Boolean(rowsError)} onClick={() => onEditingChange?.(false)}>
                {t("Done")}
              </Button>
            </SimpleTooltip>
            {isCustomSet && (
              <Button variant="outline" size="sm" onClick={onRestore}>
                {t("Restore defaults")}
              </Button>
            )}
          </>
        ) : (
          onEditingChange && (
            <Button variant="outline" onClick={() => onEditingChange(true)}>
              {t("Edit tiers")}
            </Button>
          )
        )}
      </div>
      {editing && (
        <AutoRouterAllowanceNote
          feature="tier_or_classifier_prompt"
          label={t("Custom tiers and written prompts share this allowance")}
        />
      )}
      {editing && (
        <span className="block mt-1 text-xs text-muted-foreground">
          {t(
            "Add or remove tiers to define your own set. Every custom tier needs a definition the classifier routes on, and an edited set requires the LLM or OSS classification method",
          )}
        </span>
      )}
      {editing && keywordRulesError && (
        <span className="block mt-1 text-xs text-destructive">
          {keywordRulesError}
          {t(". Edit the rules under Advanced: Keyword/Semantic Matching, or bring the tier back")}
        </span>
      )}
    </>
  );
};

const FallbackTierField: React.FC<{
  rows: readonly TierRow[];
  fallbackTierId: string;
  onValueChange: (rowId: string) => void;
}> = ({ rows, fallbackTierId, onValueChange }) => {
  const { t } = useLanguage();
  return (
    <div className="mt-4">
      <div className="flex items-center gap-2 mb-2">
        <strong className="text-base font-semibold">{t("Fallback Tier")}</strong>
        <SimpleTooltip
          content={t(
            "Where requests route when the classifier errors, times out, or returns an unparseable reply. Required for an edited tier set: the heuristic scorer cannot produce your tiers",
          )}
        >
          <Info className="size-4 text-muted-foreground" />
        </SimpleTooltip>
      </div>
      <TierRowSelect
        label={t("Fallback tier")}
        options={rows
          .filter((row) => activeTierName(row))
          .map((row) => ({ value: row.id, label: activeTierName(row) }))}
        value={fallbackTierId || null}
        onValueChange={onValueChange}
        placeholder={t("Pick the tier classifier failures route to")}
      />
    </div>
  );
};

const TierRowHeader: React.FC<{
  row: TierRow;
  index: number;
  rowCount: number;
  label: string;
  description: string | undefined;
  editing: boolean;
  isCustomSet: boolean;
  onRemove: () => void;
}> = ({ row, index, rowCount, label, description, editing, isCustomSet, onRemove }) => {
  const { t } = useLanguage();
  return (
    <div className="flex items-center gap-2 mb-2">
      <strong className="text-base font-semibold">
        {label} {t("Tier")}
      </strong>
      <SimpleTooltip
        content={
          row.definition.trim() ||
          description ||
          t("A tier you defined. The classifier routes requests matching its definition here.")
        }
      >
        <Info className="size-4 text-muted-foreground" />
      </SimpleTooltip>
      <span className="text-xs text-muted-foreground">
        {t("Tier")} {index + 1} {t("of")} {rowCount} &middot;{" "}
        {isCustomSet ? t(rowOrigin(row, isCustomSet)) : rowOrigin(row, isCustomSet)}
      </span>
      {editing && (
        <Button
          variant="ghost"
          size="sm"
          className="text-destructive hover:text-destructive/80"
          aria-label={t("Remove the {name} tier", {
            name: activeTierName(row) || t("tier {index}", { index: index + 1 }),
          })}
          disabled={rowCount <= MIN_TIER_COUNT}
          onClick={onRemove}
        >
          <Trash2 />
          {t("Remove")}
        </Button>
      )}
    </div>
  );
};

const TierRowEditFields: React.FC<{
  row: TierRow;
  index: number;
  definitionMissing: boolean;
  onPatch: (patch: Partial<Omit<TierRow, "id">>) => void;
}> = ({ row, index, definitionMissing, onPatch }) => {
  const { t } = useLanguage();
  return (
    <>
      <Input
        value={row.name}
        onChange={(event) => onPatch({ name: event.target.value })}
        placeholder={t("Tier name, e.g. SECURITY_REVIEW")}
        aria-label={t("Name for tier {index}", { index: index + 1 })}
        maxLength={MAX_TIER_NAME_CHARS}
        className="mb-2"
      />
      <Textarea
        value={row.definition}
        onChange={(event) => onPatch({ definition: event.target.value.replace(/[\r\n]+/g, " ") })}
        placeholder={
          isBuiltInTierName(row.name)
            ? t("Leave blank to keep the built-in definition")
            : t("What belongs in this tier, e.g. requests asking for a security audit")
        }
        aria-label={t("Definition for tier {index}", { index: index + 1 })}
        maxLength={MAX_TIER_DEFINITION_CHARS}
        rows={2}
        className={definitionMissing ? "mb-2 border-destructive" : "mb-2"}
      />
      {definitionMissing && (
        <span className="mb-2 block text-xs text-destructive">
          {t("A definition is required: it is the rubric the classifier routes on for this tier")}
        </span>
      )}
    </>
  );
};

export type AdaptiveEligible = "all" | "classified_tier";

export type ComplexityTierLabels = Partial<Record<keyof ComplexityTiers, string>>;

export interface ComplexityRouterConfigValue {
  tiers: ComplexityTiers;
  /** Opt into the NON_REASONING tier below SIMPLE; off keeps the four-tier ladder. */
  enable_non_reasoning_tier?: boolean;
  custom_tier_set?: CustomTierSet;
  tier_labels?: ComplexityTierLabels;
  /** An explicit pin. Unset means the default tracks the tiers - see resolveComplexityDefaultModel. */
  default_model?: string;
  classifier_type: ClassifierType;
  local_heuristic?: LocalHeuristic;
  heuristic_v2_success_threshold?: number;
  capability_classifier_config?: CapabilitySettings;
  llm_v2_config?: FuseSettings;
  classifier_llm_config?: ClassifierLLMConfig;
  jev_classifier_config?: JevClassifierConfig;
  classifier_context_window_size?: number;
  classifier_context_budget_chars?: number;
  classifier_context_per_turn_chars?: number;
  classifier_context_include_assistant_turns?: boolean;
  classifier_fallback?: ClassifierFallback;
  /** Classification instructions only; the router appends derived tier bullets after them. */
  classification_prompt?: string;
  /** Calibration examples only; the router places them after the derived tier bullets. */
  classification_examples?: string;
  /** Highest tier the scorer may decide alone under heuristic_first. Required by that type, rejected by the others. */
  heuristic_first_max_tier?: string;
  /** How near a tier boundary a score may land before hybrid defers to the classifier. Required by that type, rejected by the others. */
  hybrid_boundary_margin?: number;
  classification_mode?: ClassificationMode;
  session_affinity?: boolean;
  session_affinity_ttl_seconds?: number;
  modality_routing?: boolean;
  modality_pin_override?: boolean;
  deployment_affinity?: boolean;
  /** Plan-mode floor as a tier ROW ID, unset meaning off. The wire carries the row's name. */
  plan_mode_min_tier?: string;
  /**
   * Mid-task stall escalation. Undefined means off, which keeps all three keys out of the payload:
   * the backend rejects them alongside session pinning, user-turn classification and a custom tier
   * set, so an off router must stay silent about them rather than send an explicit false.
   */
  stall_escalation_enabled?: boolean;
  stall_escalation_window?: number;
  stall_escalation_repeat_threshold?: number;
  adaptive?: boolean;
  adaptive_weights?: AdaptiveRouterWeights;
  tier_distance_penalty?: number;
  adaptive_eligible?: AdaptiveEligible;
  return_raw_model_name?: boolean;
  /**
   * Context-window escalation gate. Undefined means untouched, which keeps both keys out of the
   * payload so the router tracks the backend defaults (enabled, 0.95 buffer); an explicit false
   * is a real opt-out and must survive the edit round-trip.
   */
  enable_context_window_escalation?: boolean;
  context_window_escalation_buffer?: number;
  cache_aware_routing?: boolean;
  cache_aware_routing_output_tokens?: number;
  cache_aware_routing_timeout_ms?: number;
  /**
   * Heuristic scorer knobs. Undefined means the operator never touched them, which keeps the key out of the
   * payload so the router tracks the backend defaults rather than freezing today's numbers.
   */
  tier_boundaries?: TierBoundaries;
  token_thresholds?: TokenThresholds;
  dimension_weights?: DimensionWeights;
  /**
   * Operator-added scoring dimensions, each carrying its own inline weight. Undefined means the router has
   * none and keeps the key out of the payload; an empty array is a real "the last row was removed" state.
   */
  custom_dimensions?: CustomDimensionRow[];
  /**
   * Score floor the reasoning-marker override must clear. Undefined keeps the key out of the payload, so the
   * floor tracks tier_boundaries.simple_medium; an explicit 0 is a real floor that promotes on the markers alone.
   */
  reasoning_override_min_score?: number;
  /**
   * Per-(tier, model) litellm_params, serialized to the sibling tier_model_configs key. The full
   * params object is held, not just reasoning_effort, so keys authored in config.yaml survive an
   * edit round-trip.
   */
  tier_model_params?: TierModelParamsByTier;
  code_keywords?: string[];
  reasoning_keywords?: string[];
  technical_keywords?: string[];
  simple_keywords?: string[];
  plan_mode_patterns?: string[];
  route_housekeeping_to_cheapest_tier?: boolean;
  housekeeping_patterns?: string[];
  reminder_markers?: ReminderMarkerPair[];
  max_tokens_from_tier_model?: boolean;
  classifier_plugin_timeout_ms?: number;
}

/** Session affinity wins where a hand-authored config sets both, matching the backend's own `or`. */
export const classificationFrequency = (value: ComplexityRouterConfigValue): ClassificationFrequency => {
  if (!value.custom_tier_set && (value.session_affinity ?? DEFAULT_SESSION_AFFINITY)) return "session";
  return value.classification_mode === "user_turn" ? "user_turn" : "every_request";
};

export const withClassificationFrequency = (
  value: ComplexityRouterConfigValue,
  frequency: ClassificationFrequency,
): ComplexityRouterConfigValue => ({
  ...value,
  classification_mode: frequency === "user_turn" ? "user_turn" : "every_request",
  session_affinity: frequency === "session",
});

interface ComplexityRouterConfigProps {
  creationFlow?: boolean;
  tierSetup?: React.ReactNode;
  modelInfo: ModelGroup[];
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  /** Parent-owned: this component unmounts when its section collapses. */
  editingTiers?: boolean;
  onEditingTiersChange?: (editing: boolean) => void;
  customTechnicalKeywords?: string[];
  onCustomTechnicalKeywordsChange?: (keywords: string[]) => void;
  // Optional: the edit-auto-router modal doesn't yet support editing keyword tier
  // rules or semantic matching, so it renders this component without them.
  keywordTierRules?: KeywordTierRule[];
  onKeywordTierRulesChange?: (rules: KeywordTierRule[]) => void;
  /** getKeywordTierRulesError's verdict, owned by the caller: importing it here would be an import cycle. */
  keywordRulesError?: string | null;
  semanticMatchingEnabled?: boolean;
  onSemanticMatchingEnabledChange?: (enabled: boolean) => void;
  embeddingModel?: string;
  onEmbeddingModelChange?: (model: string) => void;
  matchThreshold?: number;
  onMatchThresholdChange?: (threshold: number) => void;
  escalationKeywords?: string[];
  onEscalationKeywordsChange?: (keywords: string[]) => void;
  // Optional: not part of complexity_router_config, since it applies to every
  // pre-routing strategy, not just the complexity router.
  autoRouterCompression?: AutoRouterCompressionState;
  onAutoRouterCompressionChange?: (state: AutoRouterCompressionState) => void;
  showValidationErrors?: boolean;
}

export const TIER_DESCRIPTIONS: Record<
  keyof ComplexityTiers,
  { label: string; description: string; examples: string }
> = {
  NON_REASONING: {
    label: "Non-reasoning",
    description: "Operational relay work: passing information along with no judgment about it",
    examples: '"Reformat this tool output", "Acknowledge the write succeeded"',
  },
  SIMPLE: {
    label: "Simple",
    description: "Basic questions, greetings, simple factual queries",
    examples: '"Hello!", "What is Python?", "Thanks!"',
  },
  MEDIUM: {
    label: "Medium",
    description: "Standard queries requiring some reasoning or explanation",
    examples: '"Explain how REST APIs work", "Debug this error"',
  },
  COMPLEX: {
    label: "Complex",
    description: "Technical, multi-part requests requiring deep knowledge",
    examples: '"Design a microservices architecture", "Implement a rate limiter"',
  },
  REASONING: {
    label: "Reasoning",
    description: "Chain-of-thought, analysis, explicit reasoning requests",
    examples: '"Think step by step...", "Analyze the pros and cons..."',
  },
};

export const TIER_KEYS = Object.keys(TIER_DESCRIPTIONS) as Array<keyof ComplexityTiers>;

export const effectiveTierLabel = (tier: keyof ComplexityTiers, tierLabels: ComplexityTierLabels | undefined): string =>
  tierLabels?.[tier]?.trim() || TIER_DESCRIPTIONS[tier].label;

export const DEFAULT_HEURISTIC_FIRST_MAX_TIER = "SIMPLE";

/** What the Hybrid radio starts at. Required by that type, so the form always has a value to send. */
export const DEFAULT_HYBRID_BOUNDARY_MARGIN = 0.03;

/**
 * Tiers the heuristic_first threshold may name. The top tier is excluded because it would short
 * circuit every request and leave the classifier unreachable, which the backend rejects.
 */
export const HEURISTIC_FIRST_MAX_TIER_KEYS = TIER_ORDER.slice(0, -1);

const isJevCreation = (value: ComplexityRouterConfigValue, creationFlow: boolean | undefined) =>
  creationFlow && effectiveClassifierType(value) === "jev";

const CreationClassifierSettings = ({
  creationFlow,
  value,
  onChange,
  modelOptions,
  classifierEffortOptionsByModel,
  defaultModel,
  showValidationErrors,
}: Pick<ComplexityRouterConfigProps, "value" | "onChange"> & {
  creationFlow: boolean | undefined;
  modelOptions: { value: string; label: string }[];
  classifierEffortOptionsByModel: Record<string, string[] | null | undefined>;
  defaultModel: string | undefined;
  showValidationErrors: boolean;
}) => {
  const { t } = useLanguage();
  const jevCreation = isJevCreation(value, creationFlow);
  return (
    <>
      <ClassifierPrimarySettings
        section={creationFlow ? "frequency" : "all"}
        value={value}
        onChange={onChange}
        modelOptions={modelOptions}
        showValidationErrors={showValidationErrors}
      />
      {jevCreation && (
        <div className="mb-6">
          <RoutingOptions label={t("Advanced classifier settings")} showValidationErrors={showValidationErrors}>
            <div className="px-4 pb-4">
              <ClassificationMethodConfig
                advancedOnly
                jevSection="advanced"
                section="classifier"
                value={value}
                onChange={onChange}
                modelOptions={modelOptions}
                effortOptionsByModel={classifierEffortOptionsByModel}
                defaultModel={defaultModel}
                showValidationErrors={showValidationErrors}
              />
            </div>
          </RoutingOptions>
        </div>
      )}
    </>
  );
};

const ComplexityRouterConfig: React.FC<ComplexityRouterConfigProps> = ({
  creationFlow,
  tierSetup,
  modelInfo,
  value,
  onChange,
  editingTiers = false,
  onEditingTiersChange,
  customTechnicalKeywords,
  onCustomTechnicalKeywordsChange,
  keywordTierRules = [],
  onKeywordTierRulesChange,
  keywordRulesError,
  semanticMatchingEnabled = false,
  onSemanticMatchingEnabledChange,
  embeddingModel,
  onEmbeddingModelChange = () => {},
  matchThreshold = 0.5,
  onMatchThresholdChange = () => {},
  escalationKeywords = [],
  onEscalationKeywordsChange,
  autoRouterCompression = DEFAULT_AUTO_ROUTER_COMPRESSION,
  onAutoRouterCompressionChange,
  showValidationErrors = false,
}) => {
  const { t } = useLanguage();
  const forecast = isForecastClassifier(value.classifier_type);
  const tierHeading = creationFlow ? "Define your tiers" : "Models by tier";
  const customTierSet = value.custom_tier_set;
  const tierRows = activeTierRows(value);
  const tierRowsError = customTierSet ? getCustomTierRowsError(customTierSet, t) : null;
  // 只翻译内置名称，自定义层级及显示名称原样保留。
  const displayTierLabel = (row: TierRow): string => {
    const rawLabel = tierRowLabel(row, value.tier_labels);
    const customLabel = customTierSet || value.tier_labels?.[row.id as keyof ComplexityTiers]?.trim();
    return builtInTierInfo(row.id) && !customLabel ? t(rawLabel) : rawLabel;
  };

  const planModeRows = tierRows.filter((row) => row.models.length > 0);
  const planModeTierOptions = planModeRows.map((row) => ({
    value: row.id,
    label: tierRowLabel(row, value.tier_labels),
  }));
  const defaultModel = resolveComplexityDefaultModel(value, value.default_model);

  const dispatch = (action: TierSetAction) => {
    const next = applyTierSetAction(value, keywordTierRules, action);
    if (next.keywordTierRules !== keywordTierRules) onKeywordTierRulesChange?.([...next.keywordTierRules]);
    onChange(next.value);
  };

  const setRowModels = (row: TierRow, models: string[]) => dispatch({ kind: "models", id: row.id, models });
  const updateTierRow = (id: string, patch: Partial<Omit<TierRow, "id">>) => dispatch({ kind: "patch", id, patch });
  const addCustomTier = () => dispatch({ kind: "add" });
  const removeTierRow = (id: string) => dispatch({ kind: "remove", id });
  const exitToBuiltInTiers = () => dispatch({ kind: "restore" });

  const tierEffortOptionsByModel = tierEffortOptionsForModels(modelInfo);
  const fastModeByModel = Object.fromEntries(
    modelInfo.map((model) => [model.model_group, model.supports_fast_mode === true]),
  );
  const classifierEffortOptionsByModel = classifierEffortOptionsForModels(modelInfo);

  // Embedding models can't serve a chat-completion role, so they're excluded here.
  const modelOptions = modelInfo
    .filter((model) => model.mode !== "embedding")
    .map((model) => ({
      value: model.model_group,
      label: model.model_group,
    }));

  const handleTierModelParamChange = (tier: string, model: string, change: TierModelParamChange) =>
    onChange({
      ...value,
      tier_model_params: setTierModelParam(value.tier_model_params, tier, model, change),
    });

  const handleTierLabelChange = (tier: keyof ComplexityTiers, label: string) =>
    onChange({ ...value, tier_labels: { ...value.tier_labels, [tier]: label } });

  return (
    <div className="w-full max-w-none">
      <CreationClassifierSettings
        creationFlow={creationFlow}
        value={value}
        onChange={onChange}
        modelOptions={modelOptions}
        classifierEffortOptionsByModel={classifierEffortOptionsByModel}
        defaultModel={defaultModel}
        showValidationErrors={showValidationErrors}
      />
      {creationFlow && <Separator className="my-6" />}
      <div className="mb-4">
        <div className="inline-flex items-center gap-2">
          <h4 className="m-0 text-xl font-semibold text-foreground">{t(forecast ? "Solver models" : tierHeading)}</h4>
          {!forecast && (
            <SimpleTooltip
              content={t(
                "Map each complexity tier to one or more models. Simple queries use cheaper/faster models, complex queries use more capable models.",
              )}
            >
              <Info className="size-4 text-muted-foreground" />
            </SimpleTooltip>
          )}
        </div>
      </div>

      <InactiveHeuristicV2Threshold value={value} onChange={onChange} />

      {forecast ? (
        <>
          <ForecastSolverModels
            value={value}
            onChange={onChange}
            modelOptions={modelOptions}
            effortOptionsByModel={tierEffortOptionsByModel}
            fastModeByModel={fastModeByModel}
          />
          <ForecastClassifierConfig
            section="required"
            value={value}
            onChange={onChange}
            modelOptions={modelOptions}
            effortOptionsByModel={classifierEffortOptionsByModel}
          />
        </>
      ) : (
        <>
          <TierConfigIntro value={value} />
          {tierSetup}
          <Card>
            <CardContent>
              {tierRows.map((row, index) => {
                const tierInfo = builtInTierInfo(row.id);
                const label = displayTierLabel(row);
                const tierMissing = showValidationErrors && row.models.length === 0;
                const needsDefinition = !!customTierSet && !row.definition.trim() && !isBuiltInTierName(row.name);
                const definitionMissing = showValidationErrors && needsDefinition;
                const showsDisplayName = !customTierSet && !editingTiers;
                return (
                  <div key={row.id}>
                    {index > 0 && <Separator className="my-4" />}
                    <div className="mb-4">
                      <TierRowHeader
                        row={row}
                        index={index}
                        rowCount={tierRows.length}
                        label={label}
                        description={tierInfo ? t(tierInfo.description) : undefined}
                        editing={editingTiers}
                        isCustomSet={Boolean(customTierSet)}
                        onRemove={() => removeTierRow(row.id)}
                      />
                      {tierInfo && !customTierSet && (
                        <span className="block mb-2 text-xs text-muted-foreground">
                          {t("Examples:")} {tierInfo.examples}
                        </span>
                      )}
                      {editingTiers && (
                        <TierRowEditFields
                          row={row}
                          index={index}
                          definitionMissing={definitionMissing}
                          onPatch={(patch) => updateTierRow(row.id, patch)}
                        />
                      )}
                      {showsDisplayName && tierInfo && (
                        <InputGroup className="mb-2">
                          <InputGroupInput
                            value={value.tier_labels?.[row.id as keyof ComplexityTiers] ?? ""}
                            onChange={(event) =>
                              handleTierLabelChange(row.id as keyof ComplexityTiers, event.target.value)
                            }
                            placeholder={t("Display name (default: {name})", { name: t(tierInfo.label) })}
                            aria-label={t("Display name for the {name} tier", { name: t(tierInfo.label) })}
                          />
                          {value.tier_labels?.[row.id as keyof ComplexityTiers] && (
                            <InputGroupAddon align="inline-end">
                              <InputGroupButton
                                size="icon-xs"
                                aria-label={t("Clear display name for the {name} tier", { name: t(tierInfo.label) })}
                                onClick={() => handleTierLabelChange(row.id as keyof ComplexityTiers, "")}
                              >
                                <X />
                              </InputGroupButton>
                            </InputGroupAddon>
                          )}
                        </InputGroup>
                      )}
                      <MultiSelect
                        options={modelOptions}
                        value={row.models}
                        onValueChange={(models: string[]) => setRowModels(row, models)}
                        placeholder={t("Select model(s) for {tier} queries", { tier: label.toLowerCase() })}
                        emptyText={t("No models found")}
                        className={tierMissing ? "w-full border-destructive" : "w-full"}
                      />
                      <TierModelEffortRows
                        tierLabel={label}
                        models={row.models}
                        effortOptionsByModel={tierEffortOptionsByModel}
                        paramsByModel={row.params}
                        fastModeByModel={fastModeByModel}
                        onEffortChange={(model, effort) =>
                          handleTierModelParamChange(row.id, model, ["reasoning_effort", effort])
                        }
                        onFastModeChange={(model, enabled) =>
                          handleTierModelParamChange(row.id, model, ["speed", enabled ? "fast" : undefined])
                        }
                      />
                      {row.models.length > 1 && (
                        <span className="text-xs text-muted-foreground">
                          {t(
                            "Multiple models selected: the router randomly picks among them per request (or Thompson-samples within the pool when adaptive routing is on).",
                          )}
                        </span>
                      )}
                      {tierMissing && (
                        <span className="text-xs text-destructive">
                          {t("The {tier} tier is required", { tier: label })}
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}

              {!customTierSet && (
                <NonReasoningTierToggle
                  value={value}
                  onChange={onChange}
                  available={value.classifier_type === "llm" || value.classifier_type === "jev"}
                />
              )}

              <TierSetToolbar
                editing={editingTiers}
                isCustomSet={Boolean(customTierSet)}
                rowCount={tierRows.length}
                rowsError={tierRowsError}
                keywordRulesError={keywordRulesError}
                onEditingChange={onEditingTiersChange}
                onAdd={addCustomTier}
                onRestore={exitToBuiltInTiers}
              />

              {customTierSet && (
                <FallbackTierField
                  rows={tierRows}
                  fallbackTierId={customTierSet.fallback_tier_id}
                  onValueChange={(fallbackTierId) => onChange(setFallbackTier(value, fallbackTierId))}
                />
              )}
            </CardContent>
          </Card>
        </>
      )}
      <DefaultModelField value={value} onChange={onChange} modelOptions={modelOptions} />

      <RoutingOptions
        showValidationErrors={showValidationErrors}
        summary={
          { hybrid: "Hybrid local checks enabled", heuristic_first: "Heuristic first enabled" }[
            value.classifier_type as "hybrid" | "heuristic_first"
          ]
        }
      >
        {forecast && (
          <>
            <ForecastSolverModels
              additionalPoolsOnly
              value={value}
              onChange={onChange}
              modelOptions={modelOptions}
              effortOptionsByModel={tierEffortOptionsByModel}
              fastModeByModel={fastModeByModel}
            />
          </>
        )}
        <div className="rounded-lg border border-border bg-muted">
          <ComplexityRouterAdvancedSections
            hideClassifier={isJevCreation(value, creationFlow)}
            value={value}
            onChange={onChange}
            forecast={forecast}
            modelOptions={modelOptions}
            classifierEffortOptionsByModel={classifierEffortOptionsByModel}
            customTechnicalKeywords={customTechnicalKeywords}
            onCustomTechnicalKeywordsChange={onCustomTechnicalKeywordsChange}
            showValidationErrors={showValidationErrors}
            defaultModel={defaultModel}
            planModeTierOptions={planModeTierOptions}
            keywordTierRules={keywordTierRules}
            onKeywordTierRulesChange={onKeywordTierRulesChange}
            semanticMatchingEnabled={semanticMatchingEnabled}
            onSemanticMatchingEnabledChange={onSemanticMatchingEnabledChange}
            embeddingModel={embeddingModel}
            onEmbeddingModelChange={onEmbeddingModelChange}
            matchThreshold={matchThreshold}
            onMatchThresholdChange={onMatchThresholdChange}
            escalationKeywords={escalationKeywords}
            onEscalationKeywordsChange={onEscalationKeywordsChange}
            autoRouterCompression={autoRouterCompression}
            onAutoRouterCompressionChange={onAutoRouterCompressionChange}
            modelInfo={modelInfo}
            tierRows={tierRows}
            customTierSet={customTierSet}
          />
        </div>
      </RoutingOptions>
    </div>
  );
};

export default ComplexityRouterConfig;
