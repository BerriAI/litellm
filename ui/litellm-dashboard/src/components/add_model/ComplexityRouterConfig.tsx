import { SimpleTooltip } from "@/components/ui/tooltip";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ChevronRight, Info, Plus, Trash2, X } from "lucide-react";
import { Switch } from "@/components/ui/switch";

import { AffinityControls } from "./AffinityControls";
import { ModalityRoutingControls } from "./ModalityRoutingControls";
import { Card, CardContent } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
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
import AdaptiveRoutingConfig from "./AdaptiveRoutingConfig";
import ClassificationMethodConfig from "./ClassificationMethodConfig";
import ContextWindowEscalationConfig from "./ContextWindowEscalationConfig";
import { Restricted, restrictedBy } from "./TierRestrictions";
import { type TierSetAction, applyTierSetAction, setFallbackTier } from "./tier_set_actions";
import {
  REASONING_EFFORT_OPTIONS,
  ReasoningEffort,
  TierModelParamsByTier,
  setTierModelReasoningEffort,
  tierRowLabel,
} from "./complexity_router_tiers";
import TierModelEffortRows from "./TierModelEffortRows";
import EscalationKeywords from "./EscalationKeywords";
import KeywordTierRules, { KeywordTierRule } from "./KeywordTierRules";
import SemanticKeywordMatching from "./SemanticKeywordMatching";
import { type DimensionWeights, type TierBoundaries, type TokenThresholds } from "./heuristic_scoring_knobs";

export type { DimensionWeights, TierBoundaries, TokenThresholds };
export type { CustomTierSet, TierRow } from "./tier_rows";

export const DEFAULT_CLASSIFIER_TIMEOUT_MS = 3000;
export const DEFAULT_TIER_DISTANCE_PENALTY = 0.5;
export const DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE = 3;
export const DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS = 8000;
export const MIN_QUOTED_CONTEXT_TURN_CHARS = 120;
export const DEFAULT_SESSION_AFFINITY = false;
export const DEFAULT_DEPLOYMENT_AFFINITY = true;

export type ClassificationMode = "every_request" | "user_turn";

export const DEFAULT_CLASSIFICATION_MODE: ClassificationMode = "every_request";

/**
 * One operator-facing choice over the two wire fields that share the router's tier-pin machinery:
 * session affinity pins every turn, user_turn pins every turn except a new human ask.
 */
export type ClassificationFrequency = ClassificationMode | "session";

export type ComplexityTiers = {
  SIMPLE: string[];
  MEDIUM: string[];
  COMPLEX: string[];
  REASONING: string[];
};

export type ClassificationRubric = "legacy" | "agentic" | "chat" | "business";

/** What an unset preset means, matching the backend: the rubric as it shipped before calibration. */
export const DEFAULT_CLASSIFICATION_RUBRIC: ClassificationRubric = "legacy";

/**
 * Stamped on a classifier being switched on for the first time. There is no prior tier behaviour to
 * preserve at that moment, so a newly configured classifier gets the calibrated rubric while every
 * router already running an LLM classifier keeps the one it has.
 */
export const NEW_CLASSIFIER_CLASSIFICATION_RUBRIC: ClassificationRubric = "agentic";

export const CLASSIFICATION_RUBRIC_DESCRIPTIONS: Record<ClassificationRubric, { label: string; description: string }> =
  {
    legacy: {
      label: "Legacy (uncalibrated)",
      description:
        "The rubric as it shipped before calibration examples, with no worked examples at all. Routers created " +
        "before this setting existed use it, so their tier decisions and spend are unchanged. It over-routes " +
        "ordinary engineering to the most expensive tier.",
    },
    agentic: {
      label: "Agentic",
      description:
        "Anchors routine installs, builds, multi-file edits, and standard debugging at " +
        "Medium, so ordinary engineering does not route to your most expensive tier. Suits agent, terminal, and " +
        "coding-assistant traffic, and mixed traffic.",
    },
    chat: {
      label: "Chat",
      description:
        "Drops the engineering examples, for a router serving only conversational traffic that never sees those " +
        "requests.",
    },
    business: {
      label: "Business",
      description:
        "Business and sales examples plus business-oriented tier definitions: routine drafting and summarizing " +
        "stay at Medium, data-determined analysis is Complex, and only decisions under conflicting tradeoffs " +
        "reach Reasoning. Suits sales, support, and go-to-market traffic.",
    },
  };

export const CLASSIFICATION_RUBRIC_KEYS = Object.keys(CLASSIFICATION_RUBRIC_DESCRIPTIONS) as ClassificationRubric[];

export interface ClassifierLLMConfig {
  model: string;
  timeout_ms: number;
  reasoning_effort?: ReasoningEffort;
  classification_rubric?: ClassificationRubric;
  system_prompt?: string;
}

export type ClassifierType = "heuristic" | "heuristic_v2" | "llm" | "heuristic_first" | "hybrid";

/**
 * Whether this router can call classifier_llm_config.model. Mirrors the backend's
 * ComplexityRouterConfig.uses_llm_classifier, and is the single gate for every classifier-only
 * control and payload key, so a new chaining type cannot strip knobs the operator set.
 */
export const usesLlmClassifier = (classifierType: ClassifierType): boolean =>
  classifierType === "llm" || classifierType === "heuristic_first" || classifierType === "hybrid";

export type ClassifierFallback = "heuristic" | "default_model";

export const DEFAULT_CLASSIFIER_FALLBACK: ClassifierFallback = "heuristic";

export interface AdaptiveRouterWeights {
  quality: number;
  cost: number;
}

export const DEFAULT_ADAPTIVE_WEIGHTS: AdaptiveRouterWeights = { quality: 0.3, cost: 0.7 };

export type HeuristicScoringRole = "decides" | "fallback_only" | "never";

/**
 * Whether the heuristic scorer runs on this router at all, which is what gates its knobs. An LLM
 * classifier still falls back to the scorer unless the fallback is the default model, so the gate cannot be
 * a plain classifier_type check. Under heuristic_first the scorer runs first on every request and
 * decides outright whenever it lands at or below the threshold.
 */
export const heuristicScoringRoleFor = (
  classifierType: ClassifierType,
  classifierFallback: ClassifierFallback | undefined,
): HeuristicScoringRole => {
  if (classifierType === "heuristic_v2") return "never";
  if (classifierType === "heuristic" || classifierType === "heuristic_first" || classifierType === "hybrid")
    return "decides";
  return (classifierFallback ?? DEFAULT_CLASSIFIER_FALLBACK) === "heuristic" ? "fallback_only" : "never";
};

export const heuristicScoringRole = (value: ComplexityRouterConfigValue): HeuristicScoringRole =>
  value.custom_tier_set ? "never" : heuristicScoringRoleFor(value.classifier_type, value.classifier_fallback);

// Derived, never written into the value, so undoing a tier edit reverts the form with nothing left behind.
export const effectiveClassifierType = (
  value: Pick<ComplexityRouterConfigValue, "custom_tier_set" | "classifier_type">,
): ClassifierType => (value.custom_tier_set ? "llm" : value.classifier_type);

const rowOrigin = (row: TierRow, editing: boolean): string => {
  if (!editing) return row.id;
  return isBuiltInTierName(row.name) ? "built-in" : "custom";
};

const defaultModelPlaceholderFor = (derivedDefaultModel: string | undefined, isCustomSet: boolean): string => {
  if (derivedDefaultModel) return `Derived from tiers: ${derivedDefaultModel}`;
  return isCustomSet ? "Add a model to your fallback tier" : "Add a model to the Simple or Medium tier";
};

const builtInTierInfo = (rowId: string): { label: string; description: string; examples: string } | undefined => {
  const builtIn = TIER_ORDER.find((tier) => tier === rowId);
  return builtIn ? TIER_DESCRIPTIONS[builtIn] : undefined;
};

const tierConfigIntroText = (value: ComplexityRouterConfigValue): string => {
  if (value.classifier_type === "heuristic_v2") {
    return "The complexity router classifies each request with a calibrated local four-tier model (no API calls). Configure which model(s) handle each tier.";
  }
  if (heuristicScoringRole(value) === "never") {
    return "The complexity router classifies each request with your classifier model and routes it to that tier. Configure which model(s) handle each tier.";
  }
  return "The complexity router automatically classifies requests by complexity using rule-based scoring (no API calls, <1ms latency). Configure which model(s) handle each tier.";
};

const TierConfigIntro: React.FC<{ value: ComplexityRouterConfigValue }> = ({ value }) => (
  <>
    <span className="block mb-6 text-muted-foreground">{tierConfigIntroText(value)}</span>

    <span className="block mb-4 text-xs text-muted-foreground">
      {restrictedBy(value, "displayNames")?.reason ??
        "Rename a tier to use your own vocabulary in the dashboard and your spend logs. Renaming doesn't change how requests are classified, and callers never see these names."}
      {!value.custom_tier_set &&
        usesLlmClassifier(value.classifier_type) &&
        " Your classifier model reads these names, so clearer ones can sharpen its choices."}
    </span>
  </>
);

const TierSetToolbar: React.FC<{
  editing: boolean;
  isCustomSet: boolean;
  rowCount: number;
  rowsError: string | null;
  keywordRulesError: string | null | undefined;
  onEditingChange: ((editing: boolean) => void) | undefined;
  onAdd: () => void;
  onRestore: () => void;
}> = ({ editing, isCustomSet, rowCount, rowsError, keywordRulesError, onEditingChange, onAdd, onRestore }) => (
  <>
    <div className="mt-4 flex flex-wrap items-center gap-2">
      {editing ? (
        <>
          <Button variant="outline" onClick={onAdd} disabled={rowCount >= MAX_TIER_COUNT}>
            <Plus />
            Add tier
          </Button>
          <SimpleTooltip content={rowsError || undefined}>
            <Button variant="outline" disabled={Boolean(rowsError)} onClick={() => onEditingChange?.(false)}>
              Done
            </Button>
          </SimpleTooltip>
          {isCustomSet && (
            <Button variant="outline" size="sm" onClick={onRestore}>
              Restore defaults
            </Button>
          )}
        </>
      ) : (
        onEditingChange && (
          <Button variant="outline" onClick={() => onEditingChange(true)}>
            Edit tiers
          </Button>
        )
      )}
    </div>
    {editing && (
      <span className="block mt-1 text-xs text-muted-foreground">
        Add or remove tiers to define your own set. Every custom tier needs a definition the LLM classifier routes on,
        and an edited set requires the LLM classification method
      </span>
    )}
    {editing && keywordRulesError && (
      <span className="block mt-1 text-xs text-destructive">
        {keywordRulesError}. Edit the rules under Advanced: Keyword/Semantic Matching, or bring the tier back
      </span>
    )}
  </>
);

const FallbackTierField: React.FC<{
  rows: readonly TierRow[];
  fallbackTierId: string;
  onValueChange: (rowId: string) => void;
}> = ({ rows, fallbackTierId, onValueChange }) => (
  <div className="mt-4">
    <div className="flex items-center gap-2 mb-2">
      <strong className="text-base font-semibold">Fallback Tier</strong>
      <SimpleTooltip content="Where requests route when the LLM classifier errors, times out, or returns an unparseable reply. Required for an edited tier set: the heuristic scorer cannot produce your tiers.">
        <Info className="size-4 text-muted-foreground" />
      </SimpleTooltip>
    </div>
    <TierRowSelect
      label="Fallback tier"
      options={rows.filter((row) => activeTierName(row)).map((row) => ({ value: row.id, label: activeTierName(row) }))}
      value={fallbackTierId || null}
      onValueChange={onValueChange}
      placeholder="Pick the tier classifier failures route to"
    />
  </div>
);

const TierRowHeader: React.FC<{
  row: TierRow;
  index: number;
  rowCount: number;
  label: string;
  description: string | undefined;
  editing: boolean;
  isCustomSet: boolean;
  onRemove: () => void;
}> = ({ row, index, rowCount, label, description, editing, isCustomSet, onRemove }) => (
  <div className="flex items-center gap-2 mb-2">
    <strong className="text-base font-semibold">{label} Tier</strong>
    <SimpleTooltip
      content={
        row.definition.trim() ||
        description ||
        "A tier you defined. The classifier routes requests matching its definition here."
      }
    >
      <Info className="size-4 text-muted-foreground" />
    </SimpleTooltip>
    <span className="text-xs text-muted-foreground">
      Tier {index + 1} of {rowCount} &middot; {rowOrigin(row, isCustomSet)}
    </span>
    {editing && (
      <Button
        variant="ghost"
        size="sm"
        className="text-destructive hover:text-destructive/80"
        aria-label={`Remove the ${activeTierName(row) || `tier ${index + 1}`} tier`}
        disabled={rowCount <= MIN_TIER_COUNT}
        onClick={onRemove}
      >
        <Trash2 />
        Remove
      </Button>
    )}
  </div>
);

const TierRowEditFields: React.FC<{
  row: TierRow;
  index: number;
  definitionMissing: boolean;
  onPatch: (patch: Partial<Omit<TierRow, "id">>) => void;
}> = ({ row, index, definitionMissing, onPatch }) => (
  <>
    <Input
      value={row.name}
      onChange={(event) => onPatch({ name: event.target.value })}
      placeholder="Tier name, e.g. SECURITY_REVIEW"
      aria-label={`Name for tier ${index + 1}`}
      maxLength={MAX_TIER_NAME_CHARS}
      className="mb-2"
    />
    <Textarea
      value={row.definition}
      onChange={(event) => onPatch({ definition: event.target.value.replace(/[\r\n]+/g, " ") })}
      placeholder={
        isBuiltInTierName(row.name)
          ? "Leave blank to keep the built-in definition"
          : "What belongs in this tier, e.g. requests asking for a security audit"
      }
      aria-label={`Definition for tier ${index + 1}`}
      maxLength={MAX_TIER_DEFINITION_CHARS}
      rows={2}
      className={definitionMissing ? "mb-2 border-destructive" : "mb-2"}
    />
    {definitionMissing && (
      <span className="mb-2 block text-xs text-destructive">
        A definition is required: it is the rubric the classifier routes on for this tier
      </span>
    )}
  </>
);

const TierRowSelect: React.FC<{
  label: string;
  options: { value: string; label: string }[];
  value: string | null;
  onValueChange: (rowId: string) => void;
  placeholder?: string;
}> = ({ label, options, value, onValueChange, placeholder }) => (
  <Select items={options} value={value} onValueChange={(rowId: string | null) => rowId && onValueChange(rowId)}>
    <SelectTrigger aria-label={label} className="w-full">
      <SelectValue placeholder={placeholder} />
    </SelectTrigger>
    <SelectContent>
      {options.map((option) => (
        <SelectItem key={option.value} value={option.value}>
          {option.label}
        </SelectItem>
      ))}
    </SelectContent>
  </Select>
);

export type AdaptiveEligible = "all" | "classified_tier";

export type ComplexityTierLabels = Partial<Record<keyof ComplexityTiers, string>>;

export interface ComplexityRouterConfigValue {
  tiers: ComplexityTiers;
  custom_tier_set?: CustomTierSet;
  tier_labels?: ComplexityTierLabels;
  /** An explicit pin. Unset means the default tracks the tiers - see resolveComplexityDefaultModel. */
  default_model?: string;
  classifier_type: ClassifierType;
  classifier_llm_config?: ClassifierLLMConfig;
  classifier_context_window_size?: number;
  classifier_context_budget_chars?: number;
  classifier_context_per_turn_chars?: number;
  classifier_context_include_assistant_turns?: boolean;
  classifier_fallback?: ClassifierFallback;
  /** Opening instructions only; the router appends the tier bullets and the injection guard after them. */
  classification_prompt?: string;
  /** Highest tier the scorer may decide alone under heuristic_first. Required by that type, rejected by the others. */
  heuristic_first_max_tier?: string;
  /** How near a tier boundary a score may land before hybrid defers to the classifier. Required by that type, rejected by the others. */
  hybrid_boundary_margin?: number;
  classification_mode?: ClassificationMode;
  session_affinity?: boolean;
  modality_routing?: boolean;
  deployment_affinity?: boolean;
  /** Plan-mode floor as a tier ROW ID, unset meaning off. The wire carries the row's name. */
  plan_mode_min_tier?: string;
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
  /**
   * Heuristic scorer knobs. Undefined means the operator never touched them, which keeps the key out of the
   * payload so the router tracks the backend defaults rather than freezing today's numbers.
   */
  tier_boundaries?: TierBoundaries;
  token_thresholds?: TokenThresholds;
  dimension_weights?: DimensionWeights;
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
  showValidationErrors?: boolean;
}

export const TIER_DESCRIPTIONS: Record<
  keyof ComplexityTiers,
  { label: string; description: string; examples: string }
> = {
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
export const HEURISTIC_FIRST_MAX_TIER_KEYS = TIER_KEYS.slice(0, -1);

const PlanModeOverrideControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  planModeTierOptions: { value: string; label: string }[];
}> = ({ value, onChange, planModeTierOptions }) => (
  <>
    <div className="flex items-center gap-2 mb-2">
      <Switch
        checked={value.plan_mode_min_tier !== undefined}
        disabled={planModeTierOptions.length === 0}
        onCheckedChange={(enabled) =>
          onChange({
            ...value,
            plan_mode_min_tier: enabled ? planModeTierOptions.at(-1)?.value : undefined,
          })
        }
        aria-label="Route plan-mode requests to a minimum tier"
      />
      <strong className="font-semibold">Route plan-mode requests to a minimum tier</strong>
    </div>
    <span className="block text-xs mb-3 text-muted-foreground">
      Requests from coding agents in plan mode (Claude Code, GitHub Copilot) route to at least this tier. The classifier
      still wins when it picks higher, and the override only lasts while plan mode is active.
      {planModeTierOptions.length === 0 && " Add models to a tier to enable this."}
    </span>
    {value.plan_mode_min_tier !== undefined && (
      <div style={{ maxWidth: 320 }}>
        <TierRowSelect
          label="Plan-mode minimum tier"
          options={planModeTierOptions}
          value={value.plan_mode_min_tier ?? null}
          onValueChange={(tier) => onChange({ ...value, plan_mode_min_tier: tier })}
        />
      </div>
    )}
  </>
);

const ResponseFormatControls: React.FC<{
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}> = ({ value, onChange }) => (
  <>
    <div className="flex items-center gap-2 mb-2">
      <Switch
        checked={value.return_raw_model_name ?? false}
        onCheckedChange={(returnRawModelName) => onChange({ ...value, return_raw_model_name: returnRawModelName })}
        aria-label="Return raw model name"
      />
      <strong className="font-semibold">Return raw model name</strong>
    </div>
    <span className="block text-xs text-muted-foreground">
      Return the resolved underlying model name in responses instead of the autorouter alias.
    </span>
  </>
);

const ComplexityRouterConfig: React.FC<ComplexityRouterConfigProps> = ({
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
  showValidationErrors = false,
}) => {
  const customTierSet = value.custom_tier_set;
  const tierRows = activeTierRows(value);
  const tierRowsError = customTierSet ? getCustomTierRowsError(customTierSet) : null;

  const planModeRows = tierRows.filter((row) => row.models.length > 0);
  const planModeTierOptions = planModeRows.map((row) => ({
    value: row.id,
    label: tierRowLabel(row, value.tier_labels),
  }));
  const derivedDefaultModel = resolveComplexityDefaultModel(value);
  const defaultModelPlaceholder = defaultModelPlaceholderFor(derivedDefaultModel, Boolean(customTierSet));
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

  // An absent list means the proxy does not send the field yet, so every level is offered as before.
  // An empty list is the group's own answer that its deployments share no level, and is left empty.
  const effortOptionsByModel: Record<string, string[]> = Object.fromEntries(
    modelInfo.map((model) => [
      model.model_group,
      model.supported_reasoning_efforts ?? (model.supports_reasoning ? [...REASONING_EFFORT_OPTIONS] : []),
    ]),
  );

  // Embedding models can't serve a chat-completion role, so they're excluded here.
  const modelOptions = modelInfo
    .filter((model) => model.mode !== "embedding")
    .map((model) => ({
      value: model.model_group,
      label: model.model_group,
    }));

  const handleTierModelEffortChange = (tier: string, model: string, effort: ReasoningEffort | undefined) => {
    onChange({
      ...value,
      tier_model_params: setTierModelReasoningEffort(value.tier_model_params, tier, model, effort),
    });
  };

  // Clearing the select drops the key entirely rather than storing "", so an emptied pin reads as
  // "track the tiers" everywhere downstream instead of as a blank model name.
  const handleDefaultModelChange = (model: string | undefined) => {
    onChange({ ...value, default_model: model || undefined });
  };

  const handleTierLabelChange = (tier: keyof ComplexityTiers, label: string) => {
    onChange({
      ...value,
      tier_labels: { ...value.tier_labels, [tier]: label },
    });
  };

  return (
    <div className="w-full max-w-none">
      <div className="inline-flex items-center gap-2 mb-4">
        <h4 className="m-0 text-xl font-semibold text-foreground">Complexity Tier Configuration</h4>
        <SimpleTooltip content="Map each complexity tier to one or more models. Simple queries use cheaper/faster models, complex queries use more capable models.">
          <Info className="size-4 text-muted-foreground" />
        </SimpleTooltip>
      </div>

      <TierConfigIntro value={value} />

      <Card>
        <CardContent>
          {tierRows.map((row, index) => {
            const tierInfo = builtInTierInfo(row.id);
            const label = tierRowLabel(row, value.tier_labels);
            const tierMissing = showValidationErrors && row.models.length === 0;
            const needsDefinition = Boolean(customTierSet) && !row.definition.trim() && !isBuiltInTierName(row.name);
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
                    description={tierInfo?.description}
                    editing={editingTiers}
                    isCustomSet={Boolean(customTierSet)}
                    onRemove={() => removeTierRow(row.id)}
                  />
                  {tierInfo && !customTierSet && (
                    <span className="block mb-2 text-xs text-muted-foreground">Examples: {tierInfo.examples}</span>
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
                        onChange={(event) => handleTierLabelChange(row.id as keyof ComplexityTiers, event.target.value)}
                        placeholder={`Display name (default: ${tierInfo.label})`}
                        aria-label={`Display name for the ${tierInfo.label} tier`}
                      />
                      {value.tier_labels?.[row.id as keyof ComplexityTiers] && (
                        <InputGroupAddon align="inline-end">
                          <InputGroupButton
                            size="icon-xs"
                            aria-label={`Clear display name for the ${tierInfo.label} tier`}
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
                    placeholder={`Select model(s) for ${label.toLowerCase()} queries`}
                    emptyText="No models found"
                    className={tierMissing ? "w-full border-destructive" : "w-full"}
                  />
                  <TierModelEffortRows
                    tierLabel={label}
                    models={row.models}
                    effortOptionsByModel={effortOptionsByModel}
                    paramsByModel={row.params}
                    onEffortChange={(model, effort) => handleTierModelEffortChange(row.id, model, effort)}
                  />
                  {row.models.length > 1 && (
                    <span className="text-xs text-muted-foreground">
                      Multiple models selected: the router randomly picks among them per request (or Thompson-samples
                      within the pool when adaptive routing is on).
                    </span>
                  )}
                  {tierMissing && <span className="text-xs text-destructive">The {label} tier is required</span>}
                </div>
              </div>
            );
          })}

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

          <Separator className="my-4" />

          <div className="mb-2">
            <div className="flex items-center gap-2 mb-2">
              <strong className="text-base font-semibold">Default Model</strong>
              <SimpleTooltip content="Leave empty to follow the tiers. A model chosen here is pinned: it stays the default however the tiers change.">
                <Info className="size-4 text-muted-foreground" />
              </SimpleTooltip>
            </div>
            <SearchSelect
              options={modelOptions}
              value={value.default_model ?? ""}
              onValueChange={handleDefaultModelChange}
              placeholder={defaultModelPlaceholder}
              emptyText="No models found"
              aria-label="Default model"
            />
            <span className="block mt-1 text-xs text-muted-foreground">
              Used when the tier the request lands in has no model, and when the classifier fails with &quot;Route to
              the default model&quot; selected.
            </span>
          </div>
        </CardContent>
      </Card>

      <Separator className="my-6" />

      <div className="rounded-lg border border-border bg-muted">
        {[
          {
            key: "classifier",
            label: <strong className="text-foreground font-semibold">Advanced: Classification Method</strong>,
            children: (
              <ClassificationMethodConfig
                value={value}
                onChange={onChange}
                modelOptions={modelOptions}
                effortOptionsByModel={effortOptionsByModel}
                customTechnicalKeywords={customTechnicalKeywords}
                onCustomTechnicalKeywordsChange={onCustomTechnicalKeywordsChange}
                showValidationErrors={showValidationErrors}
                defaultModel={defaultModel}
              />
            ),
          },
          {
            key: "adaptive",
            label: <strong className="text-foreground font-semibold">Advanced: Adaptive Routing</strong>,
            children: (
              <Restricted by={restrictedBy(value, "adaptive")}>
                <AdaptiveRoutingConfig value={value} onChange={onChange} />
              </Restricted>
            ),
          },
          {
            key: "affinity",
            label: <strong className="text-foreground font-semibold">Advanced: Affinity</strong>,
            children: <AffinityControls value={value} onChange={onChange} />,
          },
          {
            key: "modality",
            label: <strong className="text-foreground font-semibold">Advanced: Modality Routing</strong>,
            children: <ModalityRoutingControls value={value} onChange={onChange} />,
          },
          {
            key: "plan-mode",
            label: <strong className="text-foreground font-semibold">Advanced: Plan-Mode Override</strong>,
            children: (
              <PlanModeOverrideControls value={value} onChange={onChange} planModeTierOptions={planModeTierOptions} />
            ),
          },
          {
            key: "context-window",
            label: <strong className="text-foreground font-semibold">Advanced: Context Window Escalation</strong>,
            children: <ContextWindowEscalationConfig value={value} onChange={onChange} />,
          },
          {
            key: "response",
            label: <strong className="text-foreground font-semibold">Advanced: Response Format</strong>,
            children: <ResponseFormatControls value={value} onChange={onChange} />,
          },
          ...(onEscalationKeywordsChange
            ? [
                {
                  key: "escalation",
                  label: <strong className="text-foreground font-semibold">Advanced: Escalation Keywords</strong>,
                  children: (
                    <Restricted by={restrictedBy(value, "escalation")}>
                      <EscalationKeywords keywords={escalationKeywords} onChange={onEscalationKeywordsChange} />
                    </Restricted>
                  ),
                },
              ]
            : []),
          ...(onKeywordTierRulesChange || onSemanticMatchingEnabledChange
            ? [
                {
                  key: "keyword-semantic",
                  label: <strong className="text-foreground font-semibold">Advanced: Keyword/Semantic Matching</strong>,
                  children: (
                    <>
                      {onKeywordTierRulesChange && (
                        <KeywordTierRules
                          rules={keywordTierRules}
                          onChange={onKeywordTierRulesChange}
                          tierLabels={value.tier_labels}
                          tierNames={customTierSet && tierRows.map(activeTierName).filter(Boolean)}
                        />
                      )}
                      {onKeywordTierRulesChange && onSemanticMatchingEnabledChange && <Separator className="my-4" />}
                      {onSemanticMatchingEnabledChange && (
                        <SemanticKeywordMatching
                          enabled={semanticMatchingEnabled}
                          onEnabledChange={onSemanticMatchingEnabledChange}
                          embeddingModel={embeddingModel}
                          onEmbeddingModelChange={onEmbeddingModelChange}
                          matchThreshold={matchThreshold}
                          onMatchThresholdChange={onMatchThresholdChange}
                          modelInfo={modelInfo}
                          showValidationErrors={showValidationErrors}
                        />
                      )}
                    </>
                  ),
                },
              ]
            : []),
        ].map(({ key, label, children }) => (
          <Collapsible key={key} className="border-b border-border last:border-b-0">
            <CollapsibleTrigger className="group flex w-full items-center gap-2 px-4 py-3 text-left">
              <ChevronRight className="size-4 shrink-0 text-muted-foreground transition-transform group-data-panel-open:rotate-90" />
              {label}
            </CollapsibleTrigger>
            <CollapsibleContent className="px-4 pb-4">{children}</CollapsibleContent>
          </Collapsible>
        ))}
      </div>
    </div>
  );
};

export default ComplexityRouterConfig;
