import ClassifierPrimarySettings from "./ClassifierPrimarySettings";
import { AutoRouterAllowanceNote } from "./AutoRouterAvailability";
import { transitionClassifierType } from "./classifier_type_transition";
import JevClassifierConfig from "./JevClassifierConfig";
import { Info } from "lucide-react";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Switch } from "@/components/ui/switch";
import React from "react";
import ClassifierPromptEditor from "./ClassifierPromptEditor";
import OpeningPromptEditor, { type OpeningPromptSelection } from "./OpeningPromptEditor";
import { RestrictedSection, restrictedBy } from "./TierRestrictions";
import HeuristicScoringConfig from "./HeuristicScoringConfig";
import ClassifierReasoningEffortSelect from "./ClassifierReasoningEffortSelect";
import ClassifierCircuitBreakerConfig from "./ClassifierCircuitBreakerConfig";
import ClassifierVisionConfig from "./ClassifierVisionConfig";
import { getHeuristicV2SuccessThresholdError } from "./build_complexity_router_config";
import ClassifierPluginTimeoutField from "./ClassifierPluginTimeoutField";
import ClassifierTypeRadios from "./ClassifierTypeRadios";
import type { ReasoningEffort } from "./complexity_router_tiers";
import { useComplexityScorerDefaults } from "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults";
import {
  ClassifierFallback,
  ClassifierLLMConfig,
  ClassifierType,
  ComplexityRouterConfigValue,
  DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS,
  MIN_QUOTED_CONTEXT_TURN_CHARS,
  DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE,
  DEFAULT_CLASSIFIER_FALLBACK,
  DEFAULT_CLASSIFIER_TIMEOUT_MS,
  DEFAULT_CLASSIFICATION_RUBRIC,
  ClassificationRubric,
  effectiveTierLabel,
  heuristicScoringRole,
  usesLlmClassifier,
  usesClassifierContext,
  DEFAULT_HYBRID_BOUNDARY_MARGIN,
  HEURISTIC_FIRST_MAX_TIER_KEYS,
  effectiveClassifierType,
} from "./ComplexityRouterConfig";

const DEFAULT_SCORING_EXPLANATION =
  "The router scores each request across 7 built-in dimensions: token count, code presence, reasoning markers, technical " +
  "terms, simple indicators, multi-step patterns, and question complexity, plus any custom dimensions you add. " +
  "The weighted score determines the tier:";

const HEURISTIC_V2_EXPLANATION =
  "The router estimates success probability for all four tiers with its calibrated model, then selects " +
  "the first tier that meets the success threshold. If none qualify, it selects Reasoning. " +
  "It runs locally with no classifier API call.";

const CLASSIFIER_TIMEOUT_ID = "classifier-timeout-ms";
const CLASSIFIER_CONTEXT_WINDOW_SIZE_ID = "classifier-context-window-size";
const CLASSIFIER_CONTEXT_BUDGET_CHARS_ID = "classifier-context-budget-chars";
const HYBRID_BOUNDARY_MARGIN_ID = "hybrid-boundary-margin";
const HEURISTIC_FIRST_MAX_CONTEXT_TOKENS_ID = "heuristic-first-max-context-tokens";
const HEURISTIC_V2_SUCCESS_THRESHOLD_ID = "heuristic-v2-success-threshold";

const CUSTOM_PROMPT_WITH_HEURISTIC_FALLBACK =
  "This router classifies with your own prompt, so the tier comes from whatever rubric it states. The four tier " +
  "names stay fixed. The scoring below is the heuristic, which now runs only when the classifier call fails:";

const CUSTOM_PROMPT_WITH_DEFAULT_MODEL_FALLBACK =
  "This router classifies with your own prompt, so the tier comes from whatever rubric it states. The four tier " +
  "names stay fixed. The scoring below no longer runs at all, since a failed classifier routes to the default " +
  "model instead:";

/**
 * What the scoring breakdown below it actually describes. A custom prompt means the score no longer
 * decides the tier, and pairing one with the default-model fallback means the heuristic never runs
 * at all, so the panel must not keep implying a score is involved on either router.
 */
const scoringExplanation = (value: ComplexityRouterConfigValue): string => {
  if (value.classifier_type === "heuristic_v2") return HEURISTIC_V2_EXPLANATION;
  const usesCustomPrompt =
    usesLlmClassifier(value.classifier_type) && Boolean(value.classifier_llm_config?.system_prompt?.trim());
  if (!usesCustomPrompt) return DEFAULT_SCORING_EXPLANATION;
  return value.classifier_fallback === "default_model"
    ? CUSTOM_PROMPT_WITH_DEFAULT_MODEL_FALLBACK
    : CUSTOM_PROMPT_WITH_HEURISTIC_FALLBACK;
};

/**
 * The three boundaries this card states, as displayed strings, or null until the proxy's shipped defaults
 * have arrived. Kept out of the component so the card cannot state a range the router stopped using, and
 * so the derivation does not add branches to an already dense render.
 */
const boundaryRanges = (
  shipped: Record<string, number> | undefined,
  overrides: Record<string, number> | undefined,
  reasoningOverrideMinScore: number | undefined,
): {
  simpleMedium: string;
  mediumComplex: string;
  complexReasoning: string;
  reasoningOverrideFloor: string;
} | null => {
  const effective: Record<string, number> = { ...shipped, ...overrides };
  const [low, mid, high] = [effective.simple_medium, effective.medium_complex, effective.complex_reasoning];
  if (low === undefined || mid === undefined || high === undefined) return null;
  return {
    simpleMedium: low.toFixed(2),
    mediumComplex: mid.toFixed(2),
    complexReasoning: high.toFixed(2),
    reasoningOverrideFloor: (reasoningOverrideMinScore ?? low).toFixed(2),
  };
};

const HowClassificationWorks: React.FC<{ value: ComplexityRouterConfigValue }> = ({ value }) => {
  // The shipped boundaries come from the proxy, so this card cannot state ranges the router stopped using.
  const { data: scorerDefaults, isError } = useComplexityScorerDefaults();
  const scorerRuns = heuristicScoringRole(value) !== "never";
  const ranges = boundaryRanges(
    scorerDefaults?.tier_boundaries,
    value.tier_boundaries,
    value.reasoning_override_min_score,
  );

  if (value.custom_tier_set) return null;

  return (
    <Card className="bg-muted mt-4">
      <CardContent>
        <strong className="block mb-2 font-semibold">How Classification Works</strong>
        <span className="text-[13px] text-muted-foreground">{scoringExplanation(value)}</span>
        {scorerRuns && ranges && (
          <ul className="mt-2 pl-5 text-[13px] text-muted-foreground">
            <li>
              <strong>{effectiveTierLabel("SIMPLE", value.tier_labels)}</strong>: Score &lt; {ranges.simpleMedium}
            </li>
            <li>
              <strong>{effectiveTierLabel("MEDIUM", value.tier_labels)}</strong>: Score {ranges.simpleMedium} -{" "}
              {ranges.mediumComplex}
            </li>
            <li>
              <strong>{effectiveTierLabel("COMPLEX", value.tier_labels)}</strong>: Score {ranges.mediumComplex} -{" "}
              {ranges.complexReasoning}
            </li>
            <li>
              <strong>{effectiveTierLabel("REASONING", value.tier_labels)}</strong>: Score &gt;{" "}
              {ranges.complexReasoning} (or 2+ reasoning markers with a score of at least{" "}
              {ranges.reasoningOverrideFloor})
            </li>
          </ul>
        )}
        {!ranges && isError && (
          <span className="text-[13px] block mt-2 text-muted-foreground">
            The tier score ranges could not be loaded from the proxy.
          </span>
        )}
      </CardContent>
    </Card>
  );
};

interface ClassificationMethodConfigProps {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  modelOptions: { value: string; label: string }[];
  effortOptionsByModel: Record<string, string[] | null | undefined>;
  customTechnicalKeywords?: string[];
  onCustomTechnicalKeywordsChange?: (keywords: string[]) => void;
  showValidationErrors?: boolean;
  /** The resolved default model - see resolveComplexityDefaultModel. Names and gates the radio. */
  defaultModel?: string;
  advancedOnly?: boolean;
}

export const InactiveHeuristicV2Threshold: React.FC<Pick<ClassificationMethodConfigProps, "value" | "onChange">> = ({
  value,
  onChange,
}) => {
  const threshold = value.heuristic_v2_success_threshold;
  if (effectiveClassifierType(value) === "heuristic_v2" || threshold === undefined) return null;
  const error = getHeuristicV2SuccessThresholdError(threshold);
  return (
    <section aria-label="Inactive Heuristic v2 threshold" className="mb-4 space-y-2 rounded-md border p-3">
      <p className="text-sm font-medium">
        Heuristic v2 success threshold (inactive):{" "}
        <output aria-label="Retained Heuristic v2 threshold">
          {Number.isFinite(threshold) ? threshold : "Invalid value"}
        </output>
      </p>
      <p className="text-sm text-muted-foreground">Only used when Heuristic v2 is selected</p>
      {error && (
        <p className="text-sm text-destructive" role="alert">
          {error}
        </p>
      )}
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={() => onChange({ ...value, heuristic_v2_success_threshold: undefined })}
      >
        Clear Heuristic v2 threshold
      </Button>
    </section>
  );
};

const ClassificationMethodConfig: React.FC<ClassificationMethodConfigProps> = ({
  value,
  onChange,
  modelOptions,
  effortOptionsByModel,
  customTechnicalKeywords,
  onCustomTechnicalKeywordsChange,
  showValidationErrors = false,
  defaultModel,
  advancedOnly = false,
}) => {
  const [draft, setDraft] = React.useState<{ id: string; raw: string } | null>(null);
  const hasDefaultModel = Boolean(defaultModel);
  const classifierType = effectiveClassifierType(value);
  const usesCustomPrompt = Boolean(value.classifier_llm_config?.system_prompt?.trim());
  const contextBudget = value.classifier_context_budget_chars ?? DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS;
  const contextBudgetQuotesNothing = contextBudget > 0 && contextBudget < MIN_QUOTED_CONTEXT_TURN_CHARS;
  const classificationRubric = value.classifier_llm_config?.classification_rubric ?? DEFAULT_CLASSIFICATION_RUBRIC;
  const classifierModel = value.classifier_llm_config?.model ?? "";
  const classifierReasoningEffort = value.classifier_llm_config?.reasoning_effort;
  const explicitlySupportedClassifierEfforts = effortOptionsByModel[classifierModel];
  const successThresholdError = getHeuristicV2SuccessThresholdError(value.heuristic_v2_success_threshold);
  const successThresholdDraft =
    draft?.id === HEURISTIC_V2_SUCCESS_THRESHOLD_ID &&
    Object.is(value.heuristic_v2_success_threshold, draft.raw.trim() === "" ? undefined : Number(draft.raw))
      ? draft.raw
      : null;

  const handleClassifierTypeChange = (classifierType: ClassifierType) => {
    onChange(transitionClassifierType(value, classifierType));
  };

  const handleHeuristicFirstMaxTierChange = (tier: string) => {
    onChange({ ...value, heuristic_first_max_tier: tier });
  };

  const handleHeuristicFirstMaxContextTokensChange = (raw: string) => {
    setDraft({ id: HEURISTIC_FIRST_MAX_CONTEXT_TOKENS_ID, raw });
    if (raw.trim() === "") {
      onChange({ ...value, heuristic_first_max_context_tokens: undefined });
      return;
    }
    const parsed: number = Number(raw);
    if (Number.isFinite(parsed)) {
      onChange({ ...value, heuristic_first_max_context_tokens: Math.max(1, Math.round(parsed)) });
      return;
    }
    onChange({ ...value, heuristic_first_max_context_tokens: undefined });
  };

  const handleHybridBoundaryMarginChange = (raw: string) => {
    setDraft({ id: HYBRID_BOUNDARY_MARGIN_ID, raw });
    const parsed = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(parsed)) return;
    onChange({ ...value, hybrid_boundary_margin: Math.min(1, Math.max(0, parsed)) });
  };

  const handleSuccessThresholdChange = (raw: string) => {
    setDraft({ id: HEURISTIC_V2_SUCCESS_THRESHOLD_ID, raw });
    onChange({
      ...value,
      heuristic_v2_success_threshold: raw.trim() === "" ? undefined : Number(raw),
    });
  };

  // One write for everything the prompt dialog owns. The rubric arrives here rather than through the
  // rubric handler because two onChange calls in one tick would both spread this render's `value`,
  // so whichever landed second would drop the other's edit.
  const handleClassificationPromptChange = ({
    classificationPrompt,
    classificationExamples,
    classificationRubric: selectedRubric,
  }: OpeningPromptSelection) => {
    const rubricConfig: ClassifierLLMConfig = {
      ...value.classifier_llm_config,
      model: value.classifier_llm_config?.model ?? "",
      timeout_ms: value.classifier_llm_config?.timeout_ms ?? DEFAULT_CLASSIFIER_TIMEOUT_MS,
      classification_rubric: selectedRubric,
    };
    const nextValue: ComplexityRouterConfigValue = {
      ...value,
      ...(selectedRubric && { classifier_llm_config: rubricConfig }),
      classification_prompt: classificationPrompt,
      classification_examples: classificationExamples,
    };
    onChange(nextValue);
  };

  const handleClassifierReasoningEffortChange = (reasoningEffort: ReasoningEffort | undefined) => {
    if (!value.classifier_llm_config) return;
    const { reasoning_effort: _reasoningEffort, ...classifierLlmConfig } = value.classifier_llm_config;
    onChange({
      ...value,
      classifier_llm_config:
        reasoningEffort === undefined
          ? classifierLlmConfig
          : { ...classifierLlmConfig, reasoning_effort: reasoningEffort },
    });
  };

  const handleClassifierTimeoutChange = (timeoutMs: number) => {
    onChange({
      ...value,
      classifier_llm_config: {
        ...value.classifier_llm_config,
        model: value.classifier_llm_config?.model ?? "",
        timeout_ms: timeoutMs,
      },
    });
  };

  const handleClassificationRubricChange = (classificationRubric: ClassificationRubric) => {
    onChange({
      ...value,
      classifier_llm_config: {
        ...value.classifier_llm_config,
        model: value.classifier_llm_config?.model ?? "",
        timeout_ms: value.classifier_llm_config?.timeout_ms ?? DEFAULT_CLASSIFIER_TIMEOUT_MS,
        classification_rubric: classificationRubric,
      },
    });
  };

  const handleClassifierSystemPromptChange = (systemPrompt: string | undefined) => {
    onChange({
      ...value,
      classifier_llm_config: {
        ...value.classifier_llm_config,
        model: value.classifier_llm_config?.model ?? "",
        timeout_ms: value.classifier_llm_config?.timeout_ms ?? DEFAULT_CLASSIFIER_TIMEOUT_MS,
        system_prompt: systemPrompt,
      },
    });
  };

  const handleClassifierFallbackChange = (fallback: ClassifierFallback) => {
    onChange({ ...value, classifier_fallback: fallback });
  };

  const handleClassifierContextWindowSizeChange = (windowSize: number) => {
    onChange({
      ...value,
      classifier_context_window_size: windowSize,
    });
  };

  const handleClassifierContextBudgetCharsChange = (budgetChars: number) => {
    onChange({
      ...value,
      classifier_context_budget_chars: budgetChars,
    });
  };

  const handleClassifierIntegerChange = (
    id: string,
    raw: string,
    minimum: number,
    onCommit: (value: number) => void,
  ) => {
    setDraft({ id, raw });
    const parsed = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(parsed)) return;
    onCommit(Math.max(minimum, Math.round(parsed)));
  };

  const handleClassifierContextIncludeAssistantTurnsChange = (includeAssistantTurns: boolean) => {
    onChange({
      ...value,
      classifier_context_include_assistant_turns: includeAssistantTurns,
    });
  };

  return (
    <>
      {!advancedOnly && (
        <>
          <ClassifierTypeRadios
            value={value}
            classifierType={classifierType}
            onTypeChange={handleClassifierTypeChange}
          />
          <ClassifierPrimarySettings
            value={value}
            onChange={onChange}
            modelOptions={modelOptions}
            showValidationErrors={showValidationErrors}
          />
        </>
      )}
      {advancedOnly && ["llm", "heuristic_first", "hybrid"].includes(classifierType) && (
        <div className="space-y-2">
          <Label htmlFor="auto-router-local-checks">Local checks before the judge</Label>
          <Select
            items={[
              { value: "llm", label: "Always use the judge" },
              { value: "heuristic_first", label: "Heuristic first" },
              { value: "hybrid", label: "Hybrid" },
            ]}
            value={classifierType}
            onValueChange={(next) => {
              if (next === "llm" || next === "heuristic_first" || next === "hybrid") handleClassifierTypeChange(next);
            }}
          >
            <SelectTrigger id="auto-router-local-checks" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="llm">Always use the judge</SelectItem>
              <SelectItem value="heuristic_first" disabled={Boolean(value.custom_tier_set)}>
                Heuristic first
              </SelectItem>
              <SelectItem value="hybrid" disabled={Boolean(value.custom_tier_set)}>
                Hybrid
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      )}

      {classifierType === "custom" && (
        <ClassifierPluginTimeoutField value={value} onChange={onChange} showValidationErrors={showValidationErrors} />
      )}

      {classifierType === "heuristic_v2" && (
        <div className="mt-4 space-y-2">
          <Label htmlFor={HEURISTIC_V2_SUCCESS_THRESHOLD_ID} className="block font-semibold">
            Success threshold
          </Label>
          <Input
            id={HEURISTIC_V2_SUCCESS_THRESHOLD_ID}
            type="text"
            inputMode="decimal"
            placeholder="Artifact default"
            value={successThresholdDraft ?? value.heuristic_v2_success_threshold?.toString() ?? ""}
            onChange={(event) => handleSuccessThresholdChange(event.target.value)}
            onBlur={() => {
              if (!successThresholdError) setDraft(null);
            }}
            aria-invalid={Boolean(successThresholdError)}
            aria-describedby={`${HEURISTIC_V2_SUCCESS_THRESHOLD_ID}-help${successThresholdError ? ` ${HEURISTIC_V2_SUCCESS_THRESHOLD_ID}-error` : ""}`}
          />
          <p id={`${HEURISTIC_V2_SUCCESS_THRESHOLD_ID}-help`} className="text-sm text-muted-foreground">
            Minimum predicted success probability, from 0 to 1. Higher values favor more capable tiers. Leave blank to
            use the artifact default
          </p>
          {successThresholdError && (
            <p id={`${HEURISTIC_V2_SUCCESS_THRESHOLD_ID}-error`} className="text-sm text-destructive" role="alert">
              {successThresholdError}
            </p>
          )}
        </div>
      )}

      {classifierType === "heuristic_first" && (
        <div className="mt-4 space-y-2">
          <strong className="block font-semibold">Decide locally up to</strong>
          <Select
            value={value.heuristic_first_max_tier}
            onValueChange={(tier: unknown) => handleHeuristicFirstMaxTierChange(tier as string)}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {HEURISTIC_FIRST_MAX_TIER_KEYS.map((tier) => (
                <SelectItem key={tier} value={tier}>
                  {effectiveTierLabel(tier, value.tier_labels)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-sm text-muted-foreground">
            A request the scorer places at or below this tier routes there without a classifier call. Anything the
            scorer places higher, and anything it found no signal for at all, goes to the classifier instead
          </p>
          <Label htmlFor={HEURISTIC_FIRST_MAX_CONTEXT_TOKENS_ID}>Max conversation tokens before classifier</Label>
          <Input
            id={HEURISTIC_FIRST_MAX_CONTEXT_TOKENS_ID}
            type="text"
            inputMode="numeric"
            min={1}
            aria-describedby={`${HEURISTIC_FIRST_MAX_CONTEXT_TOKENS_ID}-help`}
            value={
              draft?.id === HEURISTIC_FIRST_MAX_CONTEXT_TOKENS_ID
                ? draft.raw
                : String(value.heuristic_first_max_context_tokens ?? "")
            }
            onChange={(event) => handleHeuristicFirstMaxContextTokensChange(event.target.value)}
            onBlur={() => setDraft(null)}
            className="w-full"
          />
          <p id={`${HEURISTIC_FIRST_MAX_CONTEXT_TOKENS_ID}-help`} className="text-sm text-muted-foreground">
            Above this estimated conversation size, consult the classifier even for a short ask. Leave blank to disable
            this limit. With user-turn classification, tool continuations keep their pinned model
          </p>
        </div>
      )}

      {classifierType === "hybrid" && (
        <div className="mt-4 space-y-2">
          <strong className="block font-semibold">Boundary margin</strong>
          <Input
            id={HYBRID_BOUNDARY_MARGIN_ID}
            type="text"
            inputMode="decimal"
            value={
              draft?.id === HYBRID_BOUNDARY_MARGIN_ID
                ? draft.raw
                : String(value.hybrid_boundary_margin ?? DEFAULT_HYBRID_BOUNDARY_MARGIN)
            }
            onChange={(event) => handleHybridBoundaryMarginChange(event.target.value)}
            onBlur={() => setDraft(null)}
            className="w-full"
          />
          <p className="text-sm text-muted-foreground">
            A score further than this from every tier boundary routes on the scorer&apos;s own tier, however expensive
            that tier is. A score closer than this, and anything the scorer found no signal for at all, goes to the
            classifier to break the tie
          </p>
        </div>
      )}

      {classifierType === "jev" && <JevClassifierConfig value={value} onChange={onChange} />}
      {usesLlmClassifier(classifierType) && (
        <div className="mt-4 space-y-3">
          <ClassifierReasoningEffortSelect
            model={classifierModel}
            value={classifierReasoningEffort}
            explicitlySupported={explicitlySupportedClassifierEfforts}
            onChange={handleClassifierReasoningEffortChange}
          />
          <div>
            <Label htmlFor={CLASSIFIER_TIMEOUT_ID} className="block mb-1 font-semibold">
              Timeout (ms)
            </Label>
            <Input
              id={CLASSIFIER_TIMEOUT_ID}
              type="text"
              inputMode="numeric"
              value={
                draft?.id === CLASSIFIER_TIMEOUT_ID
                  ? draft.raw
                  : String(value.classifier_llm_config?.timeout_ms ?? DEFAULT_CLASSIFIER_TIMEOUT_MS)
              }
              onChange={(event) =>
                handleClassifierIntegerChange(
                  CLASSIFIER_TIMEOUT_ID,
                  event.target.value,
                  1,
                  handleClassifierTimeoutChange,
                )
              }
              onBlur={() => setDraft(null)}
              className="w-full"
            />
            <span className="text-xs text-muted-foreground">
              How long the classifier call has before it fails and the fallback below takes over.
            </span>
          </div>
          <ClassifierCircuitBreakerConfig
            value={value.classifier_llm_config ?? { model: "", timeout_ms: DEFAULT_CLASSIFIER_TIMEOUT_MS }}
            onChange={(classifier_llm_config) => onChange({ ...value, classifier_llm_config })}
          />
          <ClassifierVisionConfig
            value={value.classifier_llm_config ?? { model: "", timeout_ms: DEFAULT_CLASSIFIER_TIMEOUT_MS }}
            onChange={(classifier_llm_config) => onChange({ ...value, classifier_llm_config })}
          />
          <div>
            <div className="flex items-center gap-2 mb-1">
              <strong className="font-semibold">Classifier Prompt</strong>
              <SimpleTooltip content="Every rubric uses the same four tiers. They differ in the worked examples that show the classifier where the boundary between tiers sits, and the Business rubric also rewrites the tier definitions for business traffic. Pick the rubric, and write your own opening instructions and calibration examples, inside the prompt editor.">
                <Info className="size-4 text-muted-foreground" />
              </SimpleTooltip>
            </div>
            <AutoRouterAllowanceNote
              feature="tier_or_classifier_prompt"
              label="Custom instructions and examples share the custom-tier allowance"
            />
            {!value.custom_tier_set && usesCustomPrompt ? (
              <ClassifierPromptEditor
                systemPrompt={value.classifier_llm_config?.system_prompt}
                onChange={handleClassifierSystemPromptChange}
                contextWindowSize={value.classifier_context_window_size ?? DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE}
                tierLabels={value.tier_labels}
                classificationRubric={classificationRubric}
              />
            ) : (
              <OpeningPromptEditor
                classificationPrompt={value.classification_prompt}
                classificationExamples={value.classification_examples}
                onChange={handleClassificationPromptChange}
                tierSource={
                  value.custom_tier_set
                    ? { kind: "custom", tierRows: value.custom_tier_set.tiers }
                    : {
                        kind: "builtIn",
                        tierLabels: value.tier_labels,
                        classificationRubric,
                        rubricRestriction: restrictedBy(value, "classificationRubric")?.reason,
                      }
                }
                contextWindowSize={value.classifier_context_window_size ?? DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE}
              />
            )}
          </div>
        </div>
      )}
      {usesClassifierContext(classifierType) && (
        <div className="mt-4 space-y-3">
          <RestrictedSection heading="If the classifier fails" by={restrictedBy(value, "classifierFallback")}>
            <RadioGroup
              value={value.classifier_fallback ?? DEFAULT_CLASSIFIER_FALLBACK}
              onValueChange={(fallback: unknown) => handleClassifierFallbackChange(fallback as ClassifierFallback)}
            >
              <div className="inline-flex flex-col gap-2">
                <Label className="items-start font-normal leading-normal">
                  <RadioGroupItem value="heuristic" className="mt-0.5" />
                  <span>
                    <span>Score with the heuristic</span>{" "}
                    <span className="text-muted-foreground">— right when the classifier grades complexity too</span>
                  </span>
                </Label>
                <Label className="items-start font-normal leading-normal has-data-disabled:cursor-not-allowed has-data-disabled:opacity-50">
                  <RadioGroupItem value="default_model" disabled={!hasDefaultModel} className="mt-0.5" />
                  <SimpleTooltip
                    content={
                      hasDefaultModel
                        ? "Change it from the Default Model select."
                        : "Set a default model on this router to use this option"
                    }
                  >
                    <span>
                      <span>Route to the default model{defaultModel ? ` (${defaultModel})` : ""}</span>{" "}
                      <span className="text-muted-foreground">
                        — right when your prompt grades something other than complexity
                      </span>
                    </span>
                  </SimpleTooltip>
                </Label>
              </div>
            </RadioGroup>
            <span className="block text-xs text-muted-foreground">
              Applies when the classifier call errors, times out, or returns an unparseable response.
            </span>
          </RestrictedSection>
          <div>
            <Label htmlFor={CLASSIFIER_CONTEXT_WINDOW_SIZE_ID} className="block mb-1 font-semibold">
              Context Window Size
            </Label>
            <Input
              id={CLASSIFIER_CONTEXT_WINDOW_SIZE_ID}
              type="text"
              inputMode="numeric"
              value={
                draft?.id === CLASSIFIER_CONTEXT_WINDOW_SIZE_ID
                  ? draft.raw
                  : String(value.classifier_context_window_size ?? DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE)
              }
              onChange={(event) =>
                handleClassifierIntegerChange(
                  CLASSIFIER_CONTEXT_WINDOW_SIZE_ID,
                  event.target.value,
                  0,
                  handleClassifierContextWindowSizeChange,
                )
              }
              onBlur={() => setDraft(null)}
              className="w-full"
            />
            <span className="text-xs text-muted-foreground">
              Number of prior user turns sent to the classifier provider, excluding tool output and harness reminders.
              LLM and Jev default to 3 turns; Jev sends them to the configured TypeSafe endpoint. Set to 0 to omit
              conversation history. The current message and selected system text are still sent.
            </span>
          </div>
          <div>
            <Label htmlFor={CLASSIFIER_CONTEXT_BUDGET_CHARS_ID} className="block mb-1 font-semibold">
              Context Character Budget
            </Label>
            <Input
              id={CLASSIFIER_CONTEXT_BUDGET_CHARS_ID}
              type="text"
              inputMode="numeric"
              value={
                draft?.id === CLASSIFIER_CONTEXT_BUDGET_CHARS_ID
                  ? draft.raw
                  : String(value.classifier_context_budget_chars ?? DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS)
              }
              onChange={(event) =>
                handleClassifierIntegerChange(
                  CLASSIFIER_CONTEXT_BUDGET_CHARS_ID,
                  event.target.value,
                  0,
                  handleClassifierContextBudgetCharsChange,
                )
              }
              onBlur={() => setDraft(null)}
              className="w-full"
            />
            <span className="text-xs text-muted-foreground">
              Total characters of prior conversation sent to the classifier. Turns are taken newest first and quoted
              whole while they fit, so a short conversation is never cut.
            </span>
            {contextBudgetQuotesNothing && (
              <span className="block text-xs text-destructive">
                Under {MIN_QUOTED_CONTEXT_TURN_CHARS} characters there is no room to quote a turn that does not already
                fit, so a long conversation reaches the classifier with no context at all. Set Context Window Size to 0
                to turn context off deliberately.
              </span>
            )}
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Switch
                checked={value.classifier_context_include_assistant_turns ?? false}
                onCheckedChange={handleClassifierContextIncludeAssistantTurnsChange}
                size="sm"
                aria-label="Include Assistant Turns"
              />
              <strong className="font-semibold">Include Assistant Turns</strong>
              <SimpleTooltip content="Off by default. Enabling it changes tier decisions, and therefore spend, for an existing router, and sends assistant text to the classifier model, which may be a different provider than the routed model.">
                <Info className="size-4 text-muted-foreground" />
              </SimpleTooltip>
            </div>
            <span className="text-xs text-muted-foreground">
              Let the classifier read the assistant&apos;s replies, so difficulty the model stated rather than the user
              stays visible: a plan the assistant calls complex, approved with &quot;yes&quot;, is classified on the
              work being approved. Context Window Size then counts the last N turns across both roles rather than the
              last N user turns.
            </span>
          </div>
        </div>
      )}

      {heuristicScoringRole(value) !== "never" && (
        <div className="mt-4">
          <div className="flex items-center gap-2 mb-1">
            <strong className="font-semibold">Custom Technical Keywords</strong>
            <SimpleTooltip content="Domain-specific terms appended to the built-in technical keyword list. Prompts containing these terms score higher on the technical dimension and route to more capable models.">
              <Info className="size-4 text-muted-foreground" />
            </SimpleTooltip>
          </div>
          <span className="block mb-2 text-xs text-muted-foreground">
            Optional: Add terms to the built-in list to improve classification accuracy on the technical dimension.
            (e.g., udp, kafka, terraform).
          </span>
          <MultiSelect
            options={(customTechnicalKeywords ?? []).map((keyword) => ({ label: keyword, value: keyword }))}
            value={customTechnicalKeywords ?? []}
            onValueChange={(keywords: string[]) =>
              onCustomTechnicalKeywordsChange?.(
                Array.from(
                  new Set(keywords.flatMap((keyword) => keyword.split(",").map((part) => part.trim())).filter(Boolean)),
                ),
              )
            }
            placeholder="Type a keyword and press Enter"
            emptyText="Type to add a keyword"
            allowCustomValues
            className="w-full"
          />
        </div>
      )}

      {["heuristic", "heuristic_first", "hybrid"].includes(classifierType) && (
        <AutoRouterAllowanceNote feature="heuristic_tuning" label="Custom scoring rules" />
      )}
      <HeuristicScoringConfig value={value} onChange={onChange} />

      <HowClassificationWorks value={value} />
    </>
  );
};

export default ClassificationMethodConfig;
