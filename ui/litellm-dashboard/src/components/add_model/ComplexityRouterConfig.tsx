import { SimpleTooltip } from "@/components/ui/tooltip";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ChevronRight, Info, X } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { Separator } from "@/components/ui/separator";
import React from "react";
import { ModelGroup } from "@/components/llm_calls/fetch_models";
import AdaptiveRoutingConfig from "./AdaptiveRoutingConfig";
import ClassificationMethodConfig from "./ClassificationMethodConfig";
import {
  REASONING_EFFORT_OPTIONS,
  ReasoningEffort,
  TierModelParamsByTier,
  pruneTierModelParams,
  setTierModelReasoningEffort,
} from "./complexity_router_tiers";
import TierModelEffortRows from "./TierModelEffortRows";
import EscalationKeywords from "./EscalationKeywords";
import KeywordTierRules, { KeywordTierRule } from "./KeywordTierRules";
import SemanticKeywordMatching from "./SemanticKeywordMatching";
import { type DimensionWeights, type TierBoundaries, type TokenThresholds } from "./heuristic_scoring_knobs";
import { type TierRow, activeTierRows, resolveComplexityDefaultModel } from "./tier_rows";

export type { DimensionWeights, TierBoundaries, TokenThresholds };

export const DEFAULT_CLASSIFIER_TIMEOUT_MS = 3000;
export const DEFAULT_TIER_DISTANCE_PENALTY = 0.5;
export const DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE = 3;
export const DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS = 8000;
export const MIN_QUOTED_CONTEXT_TURN_CHARS = 120;
export const DEFAULT_SESSION_AFFINITY = false;
export const DEFAULT_DEPLOYMENT_AFFINITY = true;

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
  classification_rubric?: ClassificationRubric;
  system_prompt?: string;
}

export type ClassifierType = "heuristic" | "llm" | "heuristic_first";

/**
 * Whether this router can call classifier_llm_config.model. Mirrors the backend's
 * ComplexityRouterConfig.uses_llm_classifier, and is the single gate for every classifier-only
 * control and payload key, so a new chaining type cannot strip knobs the operator set.
 */
export const usesLlmClassifier = (classifierType: ClassifierType): boolean =>
  classifierType === "llm" || classifierType === "heuristic_first";

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
  if (classifierType === "heuristic" || classifierType === "heuristic_first") return "decides";
  return (classifierFallback ?? DEFAULT_CLASSIFIER_FALLBACK) === "heuristic" ? "fallback_only" : "never";
};

export const heuristicScoringRole = (value: ComplexityRouterConfigValue): HeuristicScoringRole =>
  heuristicScoringRoleFor(value.classifier_type, value.classifier_fallback);

export type AdaptiveEligible = "all" | "classified_tier";

export type ComplexityTierLabels = Partial<Record<keyof ComplexityTiers, string>>;

export interface ComplexityRouterConfigValue {
  tiers: ComplexityTiers;
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
  /** Highest tier the scorer may decide alone under heuristic_first. Required by that type, rejected by the others. */
  heuristic_first_max_tier?: string;
  session_affinity?: boolean;
  deployment_affinity?: boolean;
  /** Tier floor for coding-agent plan-mode requests. Unset means detection is off, matching the backend. */
  plan_mode_min_tier?: string;
  adaptive?: boolean;
  adaptive_weights?: AdaptiveRouterWeights;
  tier_distance_penalty?: number;
  adaptive_eligible?: AdaptiveEligible;
  return_raw_model_name?: boolean;
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

interface ComplexityRouterConfigProps {
  modelInfo: ModelGroup[];
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  customTechnicalKeywords?: string[];
  onCustomTechnicalKeywordsChange?: (keywords: string[]) => void;
  // Optional: the edit-auto-router modal doesn't yet support editing keyword tier
  // rules or semantic matching, so it renders this component without them.
  keywordTierRules?: KeywordTierRule[];
  onKeywordTierRulesChange?: (rules: KeywordTierRule[]) => void;
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

/**
 * Tiers the heuristic_first threshold may name. The top tier is excluded because it would short
 * circuit every request and leave the classifier unreachable, which the backend rejects.
 */
export const HEURISTIC_FIRST_MAX_TIER_KEYS = TIER_KEYS.slice(0, -1);

const ComplexityRouterConfig: React.FC<ComplexityRouterConfigProps> = ({
  modelInfo,
  value,
  onChange,
  customTechnicalKeywords,
  onCustomTechnicalKeywordsChange,
  keywordTierRules = [],
  onKeywordTierRulesChange,
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
  const tierRows = activeTierRows(value);
  const planModeTierOptions = tierRows
    .filter((row) => row.models.length > 0)
    .map((row) => ({ value: row.id, label: effectiveTierLabel(row.id as keyof ComplexityTiers, value.tier_labels) }));
  const derivedDefaultModel = resolveComplexityDefaultModel(value);
  const defaultModel = resolveComplexityDefaultModel(value, value.default_model);

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

  const handleTierChange = (tier: keyof ComplexityTiers, models: string[]) => {
    onChange({
      ...value,
      tiers: { ...value.tiers, [tier]: models },
      tier_model_params: pruneTierModelParams(value.tier_model_params, tier, models),
    });
  };

  const handleTierModelEffortChange = (
    tier: keyof ComplexityTiers,
    model: string,
    effort: ReasoningEffort | undefined,
  ) => {
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

      <span className="block mb-6 text-muted-foreground">
        The complexity router automatically classifies requests by complexity using rule-based scoring (no API calls,
        &lt;1ms latency). Configure which model(s) handle each tier.
      </span>

      <span className="block mb-4 text-xs text-muted-foreground">
        Rename a tier to use your own vocabulary in the dashboard and your spend logs. Renaming doesn&apos;t change how
        requests are classified, and callers never see these names.
        {usesLlmClassifier(value.classifier_type) &&
          " Your classifier model reads these names, so clearer ones can sharpen its choices."}
      </span>

      <Card>
        <CardContent>
          {tierRows.map((row: TierRow, index) => {
            const tier = row.id as keyof ComplexityTiers;
            const tierInfo = TIER_DESCRIPTIONS[tier];
            const label = effectiveTierLabel(tier, value.tier_labels);
            const tierMissing = showValidationErrors && row.models.length === 0;
            return (
              <div key={row.id}>
                {index > 0 && <Separator className="my-4" />}
                <div className="mb-4">
                  <div className="flex items-center gap-2 mb-2">
                    <strong className="text-base font-semibold">{label} Tier</strong>
                    <SimpleTooltip content={tierInfo.description}>
                      <Info className="size-4 text-muted-foreground" />
                    </SimpleTooltip>
                    <span className="text-xs text-muted-foreground">
                      Tier {index + 1} of {tierRows.length} &middot; {row.id}
                    </span>
                  </div>
                  <span className="block mb-2 text-xs text-muted-foreground">Examples: {tierInfo.examples}</span>
                  <InputGroup className="mb-2">
                    <InputGroupInput
                      value={value.tier_labels?.[tier] ?? ""}
                      onChange={(event) => handleTierLabelChange(tier, event.target.value)}
                      placeholder={`Display name (default: ${tierInfo.label})`}
                      aria-label={`Display name for the ${tierInfo.label} tier`}
                    />
                    {value.tier_labels?.[tier] && (
                      <InputGroupAddon align="inline-end">
                        <InputGroupButton
                          size="icon-xs"
                          aria-label={`Clear display name for the ${tierInfo.label} tier`}
                          onClick={() => handleTierLabelChange(tier, "")}
                        >
                          <X />
                        </InputGroupButton>
                      </InputGroupAddon>
                    )}
                  </InputGroup>
                  <MultiSelect
                    options={modelOptions}
                    value={row.models}
                    onValueChange={(models: string[]) => handleTierChange(tier, models)}
                    placeholder={`Select model(s) for ${label.toLowerCase()} queries`}
                    emptyText="No models found"
                    className={tierMissing ? "w-full border-destructive" : "w-full"}
                  />
                  <TierModelEffortRows
                    tierLabel={label}
                    models={row.models}
                    effortOptionsByModel={effortOptionsByModel}
                    paramsByModel={value.tier_model_params?.[tier]}
                    onEffortChange={(model, effort) => handleTierModelEffortChange(tier, model, effort)}
                  />
                  {row.models.length > 1 && (
                    <span className="text-xs text-muted-foreground">
                      Multiple models selected — the router randomly picks among them per request (or Thompson-samples
                      within the pool when adaptive routing is on).
                    </span>
                  )}
                  {tierMissing && <span className="text-xs text-destructive">The {label} tier is required</span>}
                </div>
              </div>
            );
          })}
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
              placeholder={
                derivedDefaultModel
                  ? `Derived from tiers: ${derivedDefaultModel}`
                  : "Add a model to the Simple or Medium tier"
              }
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
            children: <AdaptiveRoutingConfig value={value} onChange={onChange} />,
          },
          {
            key: "affinity",
            label: <strong className="text-foreground font-semibold">Advanced: Affinity</strong>,
            children: (
              <>
                <div className="flex items-center gap-2 mb-2">
                  <Switch
                    checked={value.deployment_affinity ?? DEFAULT_DEPLOYMENT_AFFINITY}
                    onCheckedChange={(deploymentAffinity) =>
                      onChange({ ...value, deployment_affinity: deploymentAffinity })
                    }
                    aria-label="Pin a session to one deployment per model group"
                  />
                  <strong className="font-semibold">Pin a session to one deployment per model group</strong>
                </div>
                <span className="block text-xs mb-3 text-muted-foreground">
                  Keeps a session on the same deployment within a group, so provider prompt caches stay warm. Turn off
                  to load-balance every turn.
                </span>
                <div className="flex items-center gap-2 mb-2">
                  <Switch
                    checked={value.session_affinity ?? DEFAULT_SESSION_AFFINITY}
                    onCheckedChange={(sessionAffinity) => onChange({ ...value, session_affinity: sessionAffinity })}
                    aria-label="Pin a session to its first model"
                  />
                  <strong className="font-semibold">Pin a session to its first model</strong>
                </div>
                <span className="block text-xs text-muted-foreground">
                  Keeps a session on its first turn&apos;s model instead of re-classifying each turn. Also pins the
                  deployment.
                </span>
              </>
            ),
          },
          {
            key: "plan-mode",
            label: <strong className="text-foreground font-semibold">Advanced: Plan-Mode Override</strong>,
            children: (
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
                  Requests from coding agents in plan mode (Claude Code, GitHub Copilot) route to at least this tier.
                  The classifier still wins when it picks higher, and the override only lasts while plan mode is active.
                  {planModeTierOptions.length === 0 && " Add models to a tier to enable this."}
                </span>
                {value.plan_mode_min_tier !== undefined && (
                  <div style={{ maxWidth: 320 }}>
                    <Select
                      items={planModeTierOptions}
                      value={value.plan_mode_min_tier}
                      onValueChange={(tier: string | null) => tier && onChange({ ...value, plan_mode_min_tier: tier })}
                    >
                      <SelectTrigger aria-label="Plan-mode minimum tier" className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {planModeTierOptions.map((option) => (
                          <SelectItem key={option.value} value={option.value}>
                            {option.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </>
            ),
          },
          {
            key: "response",
            label: <strong className="text-foreground font-semibold">Advanced: Response Format</strong>,
            children: (
              <>
                <div className="flex items-center gap-2 mb-2">
                  <Switch
                    checked={value.return_raw_model_name ?? false}
                    onCheckedChange={(returnRawModelName) =>
                      onChange({ ...value, return_raw_model_name: returnRawModelName })
                    }
                    aria-label="Return raw model name"
                  />
                  <strong className="font-semibold">Return raw model name</strong>
                </div>
                <span className="block text-xs text-muted-foreground">
                  Return the resolved underlying model name in responses instead of the autorouter alias.
                </span>
              </>
            ),
          },
          ...(onEscalationKeywordsChange
            ? [
                {
                  key: "escalation",
                  label: <strong className="text-foreground font-semibold">Advanced: Escalation Keywords</strong>,
                  children: <EscalationKeywords keywords={escalationKeywords} onChange={onEscalationKeywordsChange} />,
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
