import { SimpleTooltip } from "@/components/ui/tooltip";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ChevronRight, Info, Plus, Trash2, X } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { Separator } from "@/components/ui/separator";
import React from "react";
import { ModelGroup } from "@/components/llm_calls/fetch_models";
import AdaptiveRoutingConfig from "./AdaptiveRoutingConfig";
import ClassificationMethodConfig from "./ClassificationMethodConfig";
import { type ClassificationRubric } from "./classification_rubrics";
import { customTierDefaultModel, resolveComplexityDefaultModel, tierOptions } from "./complexity_router_tiers";
import EscalationKeywords from "./EscalationKeywords";
import KeywordTierRules, { KeywordTierRule } from "./KeywordTierRules";
import SemanticKeywordMatching from "./SemanticKeywordMatching";
import { type DimensionWeights, type TierBoundaries, type TokenThresholds } from "./heuristic_scoring_knobs";

export type { DimensionWeights, TierBoundaries, TokenThresholds };

export const DEFAULT_CLASSIFIER_TIMEOUT_MS = 3000;
export const DEFAULT_TIER_DISTANCE_PENALTY = 0.5;
export const DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE = 3;
export const DEFAULT_CLASSIFIER_CONTEXT_PER_TURN_CHARS = 200;
export const DEFAULT_SESSION_AFFINITY = false;
export const DEFAULT_DEPLOYMENT_AFFINITY = true;

export interface ComplexityTiers {
  SIMPLE: string[];
  MEDIUM: string[];
  COMPLEX: string[];
  REASONING: string[];
}

export {
  CLASSIFICATION_RUBRIC_DESCRIPTIONS,
  CLASSIFICATION_RUBRIC_KEYS,
  DEFAULT_CLASSIFICATION_RUBRIC,
  NEW_CLASSIFIER_CLASSIFICATION_RUBRIC,
} from "./classification_rubrics";
export type { ClassificationRubric } from "./classification_rubrics";

export interface ClassifierLLMConfig {
  model: string;
  timeout_ms: number;
  classification_rubric?: ClassificationRubric;
  system_prompt?: string;
}

export type ClassifierType = "heuristic" | "llm";

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
 * a plain classifier_type check.
 */
export const heuristicScoringRoleFor = (
  classifierType: ClassifierType,
  classifierFallback: ClassifierFallback | undefined,
): HeuristicScoringRole => {
  if (classifierType === "heuristic") return "decides";
  return (classifierFallback ?? DEFAULT_CLASSIFIER_FALLBACK) === "heuristic" ? "fallback_only" : "never";
};

export const heuristicScoringRole = (value: ComplexityRouterConfigValue): HeuristicScoringRole =>
  value.custom_tier_set ? "never" : heuristicScoringRoleFor(value.classifier_type, value.classifier_fallback);

export type AdaptiveEligible = "all" | "classified_tier";

export type ComplexityTierLabels = Partial<Record<keyof ComplexityTiers, string>>;

export interface TierDraft {
  /** List identity: the React key and the fallback and plan-mode pointers' target. Never serialized. */
  id: string;
  name: string;
  /** The tier's rubric bullet. Blank on a built-in name inherits the built-in criteria. */
  definition: string;
  models: string[];
}

/**
 * Present on the value when the operator edited the tier set itself. The draft IS the wire list:
 * `tiers` holds every active tier in severity order, exactly as tier_definitions will carry them,
 * so serialization and hydration are plain maps and no ordering, identity, or model placement can
 * be lost in translation. Absence means the built-in four-tier router and a payload identical to
 * before this field existed.
 */
export interface CustomTierSet {
  tiers: TierDraft[];
  fallback_tier_id: string;
}

export const isBuiltInTierName = (name: string): boolean =>
  TIER_KEYS.some((tier) => tier.toLowerCase() === name.trim().toLowerCase());

/**
 * The classifier type the payload will carry, which a custom tier set pins to "llm" without
 * writing into the value: deriving it wherever it is displayed or validated is what lets an
 * undone tier edit revert the form with nothing left behind.
 */
export const effectiveClassifierType = (
  value: Pick<ComplexityRouterConfigValue, "custom_tier_set" | "classifier_type">,
): ClassifierType => (value.custom_tier_set ? "llm" : value.classifier_type);

export const activeTierNames = (customTierSet: CustomTierSet | undefined): string[] =>
  customTierSet ? customTierSet.tiers.map((tier) => tier.name.trim()).filter(Boolean) : [...TIER_KEYS];

export interface ComplexityRouterConfigValue {
  tiers: ComplexityTiers;
  tier_labels?: ComplexityTierLabels;
  custom_tier_set?: CustomTierSet;
  /** An explicit pin. Unset means the default tracks the tiers - see resolveComplexityDefaultModel. */
  default_model?: string;
  classifier_type: ClassifierType;
  classifier_llm_config?: ClassifierLLMConfig;
  classifier_context_window_size?: number;
  classifier_context_per_turn_chars?: number;
  classifier_context_include_assistant_turns?: boolean;
  classifier_fallback?: ClassifierFallback;
  session_affinity?: boolean;
  deployment_affinity?: boolean;
  /**
   * Tier floor for coding-agent plan-mode requests, held as a tier ROW ID so renames follow
   * (built-in row ids are the four tier names, so built-in mode is id-stable by construction).
   * Serialization resolves the id to the row's name; unset means detection is off, matching the
   * backend.
   */
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

/**
 * Row IDS the plan-mode floor may point at (the backend rejects a floor whose tier has no models).
 * Ids, not names: a rename must not strand the floor, same rule as fallback_tier_id.
 */
export const planModeEligibleTiers = (tiers: ComplexityTiers, customTierSet?: CustomTierSet): string[] =>
  customTierSet
    ? customTierSet.tiers.filter((row) => row.name.trim() && row.models.length > 0).map((row) => row.id)
    : TIER_KEYS.filter((tier) => (tiers[tier] ?? []).length > 0);

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
  const planModeTiers = planModeEligibleTiers(value.tiers, value.custom_tier_set);
  const [editingTiers, setEditingTiers] = React.useState(false);
  const customTierSet = value.custom_tier_set;
  const planModeTierOptions = customTierSet
    ? customTierSet.tiers
        .filter((row) => planModeTiers.includes(row.id))
        .map((row) => ({ value: row.id, label: row.name.trim() }))
    : tierOptions(value.tier_labels).filter((option) => planModeTiers.includes(option.value));
  // An edited tier set always shows its controls: a router hydrated with custom tiers would
  // otherwise open looking read-only, with nothing hinting the set can be changed.
  const showTierControls = editingTiers || Boolean(customTierSet);
  const tierRows = customTierSet?.tiers;
  const activeCount = tierRows?.length ?? TIER_KEYS.length;
  const fallbackRow = tierRows?.find((tier) => tier.id === customTierSet?.fallback_tier_id);
  const derivedDefaultModel = customTierSet
    ? customTierDefaultModel(customTierSet)
    : resolveComplexityDefaultModel(value.tiers);
  const defaultModel = customTierSet
    ? customTierDefaultModel(customTierSet, value.default_model)
    : resolveComplexityDefaultModel(value.tiers, value.default_model);

  const builtInRow = (tier: keyof ComplexityTiers): TierDraft => ({
    id: tier,
    name: tier,
    definition: "",
    models: value.tiers[tier],
  });

  // Compares names and definitions only, deliberately not models: restoring the built-in four
  // clears the set and applyTierRows writes the rows' models back into value.tiers, so model
  // edits made inside the editor survive the mode exit instead of silently reverting.
  const isDefaultTierSet = (rows: TierDraft[]) =>
    rows.length === TIER_KEYS.length &&
    rows.every((row, index) => row.name === TIER_KEYS[index] && row.definition === "");

  // Materializes or clears the edited tier set. A set equal to the built-in four clears itself,
  // and no other value field is touched in either direction: the states a custom set forces
  // (LLM classifier, affinity and adaptive off) are derived wherever they are displayed or
  // submitted, so undoing every tier edit truly reverts the form instead of stranding forced
  // classifier state behind a cleared flag.
  const applyTierRows = (rows: TierDraft[], fallbackTierId: string) => {
    if (isDefaultTierSet(rows)) {
      const { custom_tier_set: _cleared, ...rest } = value;
      onChange({
        ...rest,
        tiers: { SIMPLE: rows[0].models, MEDIUM: rows[1].models, COMPLEX: rows[2].models, REASONING: rows[3].models },
      });
      return;
    }
    const fallback_tier_id = rows.some((row) => row.id === fallbackTierId)
      ? fallbackTierId
      : (rows.find((row) => row.name === "MEDIUM") ?? rows[0])?.id ?? "";
    onChange({ ...value, custom_tier_set: { tiers: rows, fallback_tier_id } });
  };

  const currentRows = (): [TierDraft[], string] =>
    customTierSet
      ? [customTierSet.tiers, customTierSet.fallback_tier_id]
      : [TIER_KEYS.map(builtInRow), builtInRow("MEDIUM").id];

  // Removing a built-in row snapshots its models into value.tiers (invisible on the wire while
  // the set is custom) so Restore returns the models the row had at removal, not a stale pool.
  const removeTierRow = (id: string) => {
    const [rows, fallbackId] = currentRows();
    const removed = rows.find((row) => row.id === id);
    const remaining = rows.filter((row) => row.id !== id);
    if (removed && (TIER_KEYS as string[]).includes(removed.id)) {
      const fallback_tier_id = remaining.some((row) => row.id === fallbackId)
        ? fallbackId
        : (remaining.find((row) => row.name === "MEDIUM") ?? remaining[0])?.id ?? "";
      onChange({
        ...value,
        tiers: { ...value.tiers, [removed.id]: removed.models },
        custom_tier_set: { tiers: remaining, fallback_tier_id },
      });
      return;
    }
    applyTierRows(remaining, fallbackId);
  };

  const restoreBuiltInTier = (tier: keyof ComplexityTiers) => {
    const [rows, fallbackId] = currentRows();
    const restoredInCanonicalOrder = [
      ...TIER_KEYS.flatMap((builtIn) => {
        if (builtIn === tier) return [builtInRow(tier)];
        const existing = rows.find((row) => row.id === builtIn);
        return existing ? [existing] : [];
      }),
      ...rows.filter((row) => !(TIER_KEYS as string[]).includes(row.id)),
    ];
    applyTierRows(restoredInCanonicalOrder, fallbackId);
  };

  // The id is minted against the rows themselves rather than component state: this component
  // unmounts when its section collapses while the rows live in the parent, so an instance
  // counter would reset and re-mint an id a row already holds.
  const addCustomTier = () => {
    const [rows, fallbackId] = currentRows();
    const taken = new Set(rows.map((row) => row.id));
    const id =
      Array.from({ length: rows.length + 1 }, (_, n) => `new-${n + 1}`).find((candidate) => !taken.has(candidate)) ??
      `new-${rows.length + 1}`;
    applyTierRows([...rows, { id, name: "", definition: "", models: [] }], fallbackId);
  };

  const updateTierRow = (id: string, patch: Partial<Omit<TierDraft, "id">>) => {
    const [rows, fallbackId] = currentRows();
    applyTierRows(
      rows.map((row) => (row.id === id ? { ...row, ...patch } : row)),
      fallbackId,
    );
  };

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
        {value.classifier_type === "llm" &&
          " Your classifier model reads these names, so clearer ones can sharpen its choices."}
      </span>

      <Card>
        <CardContent>
          {!customTierSet &&
            TIER_KEYS.map((tier, index) => {
              const tierInfo = TIER_DESCRIPTIONS[tier];
              const label = effectiveTierLabel(tier, value.tier_labels);
              const tierMissing = showValidationErrors && value.tiers[tier].length === 0;
              return (
                <div key={tier}>
                  {index > 0 && <Separator className="my-4" />}
                  <div className="mb-4">
                    <div className="flex items-center gap-2 mb-2">
                      <strong className="text-base font-semibold">{label} Tier</strong>
                      <SimpleTooltip content={tierInfo.description}>
                        <Info className="size-4 text-muted-foreground" />
                      </SimpleTooltip>
                      <span className="text-xs text-muted-foreground">
                        Tier {index + 1} of {activeCount} &middot; {tier}
                      </span>
                      {showTierControls && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-destructive hover:text-destructive/80"
                          aria-label={`Remove the ${tier} tier`}
                          disabled={activeCount <= 2}
                          onClick={() => removeTierRow(tier)}
                        >
                          <Trash2 />
                          Remove
                        </Button>
                      )}
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
                      value={value.tiers[tier]}
                      onValueChange={(models: string[]) => handleTierChange(tier, models)}
                      placeholder={`Select model(s) for ${label.toLowerCase()} queries`}
                      emptyText="No models found"
                      className={tierMissing ? "w-full border-destructive" : "w-full"}
                    />
                    {value.tiers[tier].length > 1 && (
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
          {tierRows?.map((row, index) => {
            const rowName = row.name.trim();
            const builtIn = isBuiltInTierName(rowName);
            const builtInInfo = builtIn ? TIER_DESCRIPTIONS[rowName.toUpperCase() as keyof ComplexityTiers] : undefined;
            const nameMissing = showValidationErrors && !rowName;
            const definitionMissing = showValidationErrors && !row.definition.trim() && !builtIn;
            const modelsMissing = showValidationErrors && row.models.length === 0;
            return (
              <div key={row.id}>
                {index > 0 && <Separator className="my-4" />}
                <div className="mb-4">
                  <div className="flex items-center gap-2 mb-2">
                    <strong className="text-base font-semibold">{rowName || "New"} Tier</strong>
                    <SimpleTooltip
                      content={
                        builtInInfo?.description ??
                        "A tier you defined. The classifier routes here when a request matches the definition below."
                      }
                    >
                      <Info className="size-4 text-muted-foreground" />
                    </SimpleTooltip>
                    <span className="text-xs text-muted-foreground">
                      Tier {index + 1} of {activeCount} &middot; {builtIn ? "built-in" : "custom"}
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-destructive hover:text-destructive/80"
                      aria-label={`Remove the ${rowName || `tier ${index + 1}`} tier`}
                      disabled={activeCount <= 2}
                      onClick={() => removeTierRow(row.id)}
                    >
                      <Trash2 />
                      Remove
                    </Button>
                  </div>
                  {builtInInfo && (
                    <span className="block mb-2 text-xs text-muted-foreground">Examples: {builtInInfo.examples}</span>
                  )}
                  <Input
                    value={row.name}
                    onChange={(event) => updateTierRow(row.id, { name: event.target.value })}
                    placeholder="Tier name, e.g. SECURITY_REVIEW"
                    aria-label={`Name for tier ${index + 1}`}
                    className={nameMissing ? "mb-2 border-destructive" : "mb-2"}
                  />
                  <Textarea
                    value={row.definition}
                    onChange={(event) => updateTierRow(row.id, { definition: event.target.value })}
                    placeholder={
                      builtIn
                        ? "Leave blank to keep the built-in definition the classifier already uses for this tier"
                        : "What belongs in this tier. The LLM classifier reads this definition to decide when a request routes here, e.g. requests asking for a security audit, vulnerability review, or exploit analysis"
                    }
                    aria-label={`Definition for tier ${index + 1}`}
                    rows={2}
                    className={definitionMissing ? "mb-2 border-destructive" : "mb-2"}
                  />
                  {definitionMissing && (
                    <span className="block mb-2 text-xs text-destructive">
                      A definition is required: it is the rubric the classifier uses for this tier
                    </span>
                  )}
                  <MultiSelect
                    options={modelOptions}
                    value={row.models}
                    onValueChange={(models: string[]) => updateTierRow(row.id, { models })}
                    placeholder={`Select model(s) for the ${rowName || "new"} tier`}
                    aria-label={`Models for tier ${index + 1}`}
                    emptyText="No models found"
                    className={modelsMissing ? "w-full border-destructive" : "w-full"}
                  />
                  {modelsMissing && (
                    <span className="text-xs text-destructive">Select at least one model for this tier</span>
                  )}
                </div>
              </div>
            );
          })}
          <Separator className="my-4" />

          <div className="mb-4">
            <div className="flex flex-wrap items-center gap-2">
              {showTierControls ? (
                <>
                  <Button variant="outline" onClick={addCustomTier} disabled={activeCount >= 8}>
                    <Plus />
                    Add tier
                  </Button>
                  {editingTiers && (
                    <Button variant="outline" onClick={() => setEditingTiers(false)}>
                      Done
                    </Button>
                  )}
                  {TIER_KEYS.filter((tier) => customTierSet && !tierRows?.some((row) => row.id === tier)).map(
                    (tier) => (
                      <Button key={tier} variant="outline" size="sm" onClick={() => restoreBuiltInTier(tier)}>
                        Restore {tier}
                      </Button>
                    ),
                  )}
                </>
              ) : (
                <Button variant="outline" onClick={() => setEditingTiers(true)}>
                  Edit tiers
                </Button>
              )}
            </div>
            {showTierControls && (
              <span className="block mt-1 text-xs text-muted-foreground">
                Add or remove tiers to define your own tier set. Every new tier needs a definition the LLM classifier
                uses to route to it; editing the set requires the LLM classification method and disables escalation,
                adaptive selection, session pinning, and display names.
              </span>
            )}
          </div>
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
          {customTierSet && (
            <div className="mb-2">
              <Separator className="my-4" />
              <div className="flex items-center gap-2 mb-2">
                <strong className="text-base font-semibold">Fallback Tier</strong>
                <SimpleTooltip content="Where requests route when the LLM classifier errors, times out, or returns an unparseable reply. Required for an edited tier set: the heuristic scorer cannot produce your tiers.">
                  <Info className="size-4 text-muted-foreground" />
                </SimpleTooltip>
              </div>
              <Select
                items={customTierSet.tiers
                  .filter((row) => row.name.trim())
                  .map((row) => ({ value: row.id, label: row.name.trim() }))}
                value={fallbackRow?.id ?? null}
                onValueChange={(fallbackTierId: string | null) =>
                  fallbackTierId && applyTierRows(customTierSet.tiers, fallbackTierId)
                }
              >
                <SelectTrigger
                  aria-label="Fallback tier"
                  className={showValidationErrors && !fallbackRow ? "w-full border-destructive" : "w-full"}
                >
                  <SelectValue placeholder="Pick the tier classifier failures route to" />
                </SelectTrigger>
                <SelectContent>
                  {customTierSet.tiers
                    .filter((row) => row.name.trim())
                    .map((row) => (
                      <SelectItem key={row.id} value={row.id}>
                        {row.name.trim()}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>
          )}
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
            children: customTierSet ? (
              <span className="text-sm text-muted-foreground">
                Adaptive routing is unavailable with an edited tier set: it scores models along the built-in tier
                ladder, which your tier set replaces.
              </span>
            ) : (
              <AdaptiveRoutingConfig value={value} onChange={onChange} />
            ),
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
                    checked={customTierSet ? false : value.session_affinity ?? DEFAULT_SESSION_AFFINITY}
                    onCheckedChange={(sessionAffinity) => onChange({ ...value, session_affinity: sessionAffinity })}
                    aria-label="Pin a session to its first model"
                    disabled={Boolean(customTierSet)}
                  />
                  <strong className="font-semibold">Pin a session to its first model</strong>
                </div>
                <span className="block text-xs text-muted-foreground">
                  {customTierSet
                    ? "Unavailable with an edited tier set: escalating a pinned session walks the built-in tier ladder, which your tier set replaces."
                    : "Keeps a session on its first turn's model instead of re-classifying each turn. Also pins the deployment."}
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
                    disabled={planModeTiers.length === 0}
                    onCheckedChange={(enabled) =>
                      onChange({ ...value, plan_mode_min_tier: enabled ? planModeTiers.at(-1) : undefined })
                    }
                    aria-label="Route plan-mode requests to a minimum tier"
                  />
                  <strong className="font-semibold">Route plan-mode requests to a minimum tier</strong>
                </div>
                <span className="block text-xs mb-3 text-muted-foreground">
                  Requests from coding agents in plan mode (Claude Code, GitHub Copilot) route to at least this tier.
                  The classifier still wins when it picks higher, and the override only lasts while plan mode is active.
                  {planModeTiers.length === 0 && " Add models to a tier to enable this."}
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
                  children: customTierSet ? (
                    <span className="text-sm text-muted-foreground">
                      Escalation keywords are unavailable with an edited tier set: they bump requests along the built-in
                      tier ladder, which your tier set replaces.
                    </span>
                  ) : (
                    <EscalationKeywords keywords={escalationKeywords} onChange={onEscalationKeywordsChange} />
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
                          tierLabels={customTierSet ? undefined : value.tier_labels}
                          tierNames={customTierSet ? activeTierNames(customTierSet) : undefined}
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
