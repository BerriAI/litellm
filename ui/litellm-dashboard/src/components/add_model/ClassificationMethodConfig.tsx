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
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
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
import { ClassifierMenu, ClassifierOption } from "./AutoRouterClassifierTabs";
import { isHeuristicChain } from "./classifier_types";
import HeuristicKeywordOverrides from "./HeuristicKeywordOverrides";
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
  effectiveTierDisplayLabel,
  heuristicScoringRole,
  heuristicTuningType,
  usesHeuristicV2,
  usesLlmClassifier,
  usesClassifierContext,
  DEFAULT_HYBRID_BOUNDARY_MARGIN,
  HEURISTIC_FIRST_MAX_TIER_KEYS,
  effectiveClassifierType,
} from "./ComplexityRouterConfig";

const DEFAULT_SCORING_EXPLANATION: ParseKeys = "addModel.classificationMethodConfig.explanations.defaultScoring";

const HEURISTIC_V2_EXPLANATION: ParseKeys = "addModel.classificationMethodConfig.explanations.heuristicV2";

const CLASSIFIER_TIMEOUT_ID = "classifier-timeout-ms";
const CLASSIFIER_CONTEXT_WINDOW_SIZE_ID = "classifier-context-window-size";
const CLASSIFIER_CONTEXT_BUDGET_CHARS_ID = "classifier-context-budget-chars";
const HYBRID_BOUNDARY_MARGIN_ID = "hybrid-boundary-margin";
const HEURISTIC_V2_SUCCESS_THRESHOLD_ID = "heuristic-v2-success-threshold";

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
  const { t } = useTranslation();
  const { data: scorerDefaults, isError } = useComplexityScorerDefaults();
  const scorerRuns = heuristicScoringRole(value) !== "never";
  const ranges = boundaryRanges(
    scorerDefaults?.tier_boundaries,
    value.tier_boundaries,
    value.reasoning_override_min_score,
  );
  const scorerDefaultsUnavailable = scorerRuns && !ranges && isError;

  if (value.custom_tier_set) return null;

  return (
    <Card className="bg-muted mt-4">
      <CardContent>
        <strong className="block mb-2 font-semibold">
          {t("addModel.classificationMethodConfig.howItWorks.title")}
        </strong>
        <span className="text-[13px] text-muted-foreground">
          {usesHeuristicV2(value) ? t(HEURISTIC_V2_EXPLANATION) : t(DEFAULT_SCORING_EXPLANATION)}
          {usesHeuristicV2(value) &&
            (isHeuristicChain(value.classifier_type)
              ? t("addModel.classificationMethodConfig.howItWorks.judgeChoosesTier")
              : t("addModel.classificationMethodConfig.howItWorks.selectsReasoning"))}
        </span>
        {scorerRuns && ranges && (
          <ul className="mt-2 pl-5 text-[13px] text-muted-foreground">
            <li>
              <strong>{effectiveTierDisplayLabel("SIMPLE", value.tier_labels, t)}</strong>:{" "}
              {t("addModel.classificationMethodConfig.howItWorks.scoreBelow", { max: ranges.simpleMedium })}
            </li>
            <li>
              <strong>{effectiveTierDisplayLabel("MEDIUM", value.tier_labels, t)}</strong>:{" "}
              {t("addModel.classificationMethodConfig.howItWorks.scoreBetween", {
                min: ranges.simpleMedium,
                max: ranges.mediumComplex,
              })}
            </li>
            <li>
              <strong>{effectiveTierDisplayLabel("COMPLEX", value.tier_labels, t)}</strong>:{" "}
              {t("addModel.classificationMethodConfig.howItWorks.scoreBetween", {
                min: ranges.mediumComplex,
                max: ranges.complexReasoning,
              })}
            </li>
            <li>
              <strong>{effectiveTierDisplayLabel("REASONING", value.tier_labels, t)}</strong>:{" "}
              {t("addModel.classificationMethodConfig.howItWorks.scoreAbove", {
                min: ranges.complexReasoning,
                floor: ranges.reasoningOverrideFloor,
              })}
            </li>
          </ul>
        )}
        {scorerDefaultsUnavailable && (
          <span className="text-[13px] block mt-2 text-muted-foreground">
            {t("addModel.classificationMethodConfig.howItWorks.rangesUnavailable")}
          </span>
        )}
      </CardContent>
    </Card>
  );
};

interface ClassificationMethodConfigProps {
  jevSection?: "all" | "advanced";
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
  section?: "selection" | "classifier" | "heuristic";
}

export const InactiveHeuristicV2Threshold: React.FC<Pick<ClassificationMethodConfigProps, "value" | "onChange">> = ({
  value,
  onChange,
}) => {
  const { t } = useTranslation();
  const threshold = value.heuristic_v2_success_threshold;
  if (usesHeuristicV2(value) || threshold === undefined) return null;
  const error = getHeuristicV2SuccessThresholdError(threshold);
  if (effectiveClassifierType(value) === "llm" && !error) return null;
  return (
    <section
      aria-label={t("addModel.classificationMethodConfig.inactiveThreshold.sectionLabel")}
      className="mb-4 space-y-2 rounded-md border p-3"
    >
      <p className="text-sm font-medium">
        {t("addModel.classificationMethodConfig.inactiveThreshold.title")}{" "}
        <output aria-label={t("addModel.classificationMethodConfig.inactiveThreshold.valueLabel")}>
          {Number.isFinite(threshold)
            ? threshold
            : t("addModel.classificationMethodConfig.inactiveThreshold.invalidValue")}
        </output>
      </p>
      <p className="text-sm text-muted-foreground">{t("addModel.classificationMethodConfig.inactiveThreshold.hint")}</p>
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
        {t("addModel.classificationMethodConfig.inactiveThreshold.clear")}
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
  jevSection,
  section,
}) => {
  const { t } = useTranslation();
  const [draft, setDraft] = React.useState<{ id: string; raw: string } | null>(null);
  const hasDefaultModel = Boolean(defaultModel);
  const classifierType = effectiveClassifierType(value);
  const showSelection = section === undefined || section === "selection";
  const showClassifier = section === undefined || section === "classifier";
  const heuristic = heuristicTuningType(value);
  const showHeuristic = (section === undefined || section === "heuristic") && heuristic !== undefined;
  const showLocalChecks =
    showSelection && advancedOnly && ["llm", "heuristic_first", "hybrid"].includes(classifierType);
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
      {!advancedOnly && showSelection && (
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
      {showLocalChecks && (
        <div className="space-y-2">
          <Label htmlFor="auto-router-local-checks">{t("addModel.classificationMethodConfig.localChecks.label")}</Label>
          <Select
            items={[
              { value: "llm", label: t("addModel.classificationMethodConfig.localChecks.alwaysJudge") },
              { value: "heuristic_first", label: t("addModel.classificationMethodConfig.localChecks.heuristicFirst") },
              { value: "hybrid", label: t("addModel.classificationMethodConfig.localChecks.hybrid") },
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
              <SelectItem value="llm">{t("addModel.classificationMethodConfig.localChecks.alwaysJudge")}</SelectItem>
              <SelectItem value="heuristic_first" disabled={Boolean(value.custom_tier_set)}>
                {t("addModel.classificationMethodConfig.localChecks.heuristicFirst")}
              </SelectItem>
              <SelectItem value="hybrid" disabled={Boolean(value.custom_tier_set)}>
                {t("addModel.classificationMethodConfig.localChecks.hybrid")}
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      )}

      {showSelection && isHeuristicChain(classifierType) && (
        <div className="mt-4 space-y-2">
          <Label htmlFor="auto-router-local-heuristic">
            {t("addModel.classificationMethodConfig.localHeuristic.label")}
          </Label>
          <ClassifierMenu
            id="auto-router-local-heuristic"
            label={t("addModel.classificationMethodConfig.localHeuristic.label")}
            selectedLabel={
              value.local_heuristic === "heuristic_v2"
                ? t("addModel.classificationMethodConfig.localHeuristic.v2")
                : t("addModel.classificationMethodConfig.localHeuristic.v1")
            }
            feature={value.local_heuristic === "heuristic_v2" ? "heuristic_v2" : undefined}
            value={value.local_heuristic ?? "heuristic"}
            onValueChange={(next) => {
              if (next === "heuristic" || next === "heuristic_v2") onChange({ ...value, local_heuristic: next });
            }}
          >
            <ClassifierOption
              value="heuristic"
              label={t("addModel.classificationMethodConfig.localHeuristic.v1")}
              description={t("addModel.classificationMethodConfig.localHeuristic.v1Description")}
            />
            <ClassifierOption
              value="heuristic_v2"
              label={t("addModel.classificationMethodConfig.localHeuristic.v2")}
              description={t("addModel.classificationMethodConfig.localHeuristic.v2Description")}
              feature="heuristic_v2"
            />
          </ClassifierMenu>
        </div>
      )}

      {showClassifier && classifierType === "custom" && (
        <ClassifierPluginTimeoutField value={value} onChange={onChange} showValidationErrors={showValidationErrors} />
      )}

      {showHeuristic && heuristic === "heuristic_v2" && (
        <div className="mt-4 space-y-2">
          <Label htmlFor={HEURISTIC_V2_SUCCESS_THRESHOLD_ID} className="block font-semibold">
            {t("addModel.classificationMethodConfig.successThreshold.label")}
          </Label>
          <Input
            id={HEURISTIC_V2_SUCCESS_THRESHOLD_ID}
            type="text"
            inputMode="decimal"
            placeholder={t("addModel.classificationMethodConfig.successThreshold.placeholder")}
            value={successThresholdDraft ?? value.heuristic_v2_success_threshold?.toString() ?? ""}
            onChange={(event) => handleSuccessThresholdChange(event.target.value)}
            onBlur={() => {
              if (!successThresholdError) setDraft(null);
            }}
            aria-invalid={Boolean(successThresholdError)}
            aria-describedby={`${HEURISTIC_V2_SUCCESS_THRESHOLD_ID}-help${successThresholdError ? ` ${HEURISTIC_V2_SUCCESS_THRESHOLD_ID}-error` : ""}`}
          />
          <p id={`${HEURISTIC_V2_SUCCESS_THRESHOLD_ID}-help`} className="text-sm text-muted-foreground">
            {t("addModel.classificationMethodConfig.successThreshold.help")}
          </p>
          {successThresholdError && (
            <p id={`${HEURISTIC_V2_SUCCESS_THRESHOLD_ID}-error`} className="text-sm text-destructive" role="alert">
              {successThresholdError}
            </p>
          )}
        </div>
      )}

      {showSelection && classifierType === "heuristic_first" && (
        <div className="mt-4 space-y-2">
          <Label htmlFor="auto-router-local-max-tier" className="block font-semibold">
            {t("addModel.classificationMethodConfig.decideLocally.label")}
          </Label>
          <Select
            value={value.heuristic_first_max_tier}
            onValueChange={(tier: unknown) => handleHeuristicFirstMaxTierChange(tier as string)}
          >
            <SelectTrigger id="auto-router-local-max-tier" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {HEURISTIC_FIRST_MAX_TIER_KEYS.map((tier) => (
                <SelectItem key={tier} value={tier}>
                  {effectiveTierDisplayLabel(tier, value.tier_labels, t)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-sm text-muted-foreground">
            {usesHeuristicV2(value)
              ? t("addModel.classificationMethodConfig.decideLocally.helpV2")
              : t("addModel.classificationMethodConfig.decideLocally.helpV1")}
          </p>
        </div>
      )}

      {showSelection && classifierType === "hybrid" && (
        <div className="mt-4 space-y-2">
          <Label htmlFor={HYBRID_BOUNDARY_MARGIN_ID} className="block font-semibold">
            {usesHeuristicV2(value)
              ? t("addModel.classificationMethodConfig.hybridMargin.labelV2")
              : t("addModel.classificationMethodConfig.hybridMargin.labelV1")}
          </Label>
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
            {usesHeuristicV2(value)
              ? t("addModel.classificationMethodConfig.hybridMargin.helpV2")
              : t("addModel.classificationMethodConfig.hybridMargin.helpV1")}
          </p>
        </div>
      )}

      {showClassifier && classifierType === "jev" && (
        <JevClassifierConfig value={value} onChange={onChange} section={jevSection} />
      )}
      {showClassifier && usesLlmClassifier(classifierType) && (
        <div className="mt-4 space-y-3">
          <ClassifierReasoningEffortSelect
            model={classifierModel}
            value={classifierReasoningEffort}
            explicitlySupported={explicitlySupportedClassifierEfforts}
            onChange={handleClassifierReasoningEffortChange}
          />
          <div>
            <Label htmlFor={CLASSIFIER_TIMEOUT_ID} className="block mb-1 font-semibold">
              {t("addModel.classificationMethodConfig.timeout.label")}
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
              {t("addModel.classificationMethodConfig.timeout.help")}
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
              <strong className="font-semibold">
                {t("addModel.classificationMethodConfig.classifierPrompt.title")}
              </strong>
              <SimpleTooltip content={t("addModel.classificationMethodConfig.classifierPrompt.tooltip")}>
                <Info className="size-4 text-muted-foreground" />
              </SimpleTooltip>
            </div>
            <AutoRouterAllowanceNote
              feature="tier_or_classifier_prompt"
              label={t("addModel.classificationMethodConfig.classifierPrompt.allowance")}
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
      {showClassifier && usesClassifierContext(classifierType) && (
        <div className="mt-4 space-y-3">
          <RestrictedSection
            heading={t("addModel.classificationMethodConfig.fallback.heading")}
            by={restrictedBy(value, "classifierFallback")}
          >
            <RadioGroup
              value={value.classifier_fallback ?? DEFAULT_CLASSIFIER_FALLBACK}
              onValueChange={(fallback: unknown) => handleClassifierFallbackChange(fallback as ClassifierFallback)}
            >
              <div className="inline-flex flex-col gap-2">
                <Label className="items-start font-normal leading-normal">
                  <RadioGroupItem value="heuristic" className="mt-0.5" />
                  <span>
                    <span>{t("addModel.classificationMethodConfig.fallback.heuristic")}</span>{" "}
                    <span className="text-muted-foreground">
                      {t("addModel.classificationMethodConfig.fallback.heuristicHint")}
                    </span>
                  </span>
                </Label>
                <Label className="items-start font-normal leading-normal has-data-disabled:cursor-not-allowed has-data-disabled:opacity-50">
                  <RadioGroupItem value="default_model" disabled={!hasDefaultModel} className="mt-0.5" />
                  <SimpleTooltip
                    content={
                      hasDefaultModel
                        ? t("addModel.classificationMethodConfig.fallback.defaultModelSetTooltip")
                        : t("addModel.classificationMethodConfig.fallback.defaultModelMissingTooltip")
                    }
                  >
                    <span>
                      <span>
                        {t("addModel.classificationMethodConfig.fallback.defaultModel")}
                        {defaultModel ? ` (${defaultModel})` : ""}
                      </span>{" "}
                      <span className="text-muted-foreground">
                        {t("addModel.classificationMethodConfig.fallback.defaultModelHint")}
                      </span>
                    </span>
                  </SimpleTooltip>
                </Label>
              </div>
            </RadioGroup>
            <span className="block text-xs text-muted-foreground">
              {t("addModel.classificationMethodConfig.fallback.help")}
            </span>
          </RestrictedSection>
          <div>
            <Label htmlFor={CLASSIFIER_CONTEXT_WINDOW_SIZE_ID} className="block mb-1 font-semibold">
              {t("addModel.classificationMethodConfig.contextWindow.label")}
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
              {t("addModel.classificationMethodConfig.contextWindow.help")}
            </span>
          </div>
          <div>
            <Label htmlFor={CLASSIFIER_CONTEXT_BUDGET_CHARS_ID} className="block mb-1 font-semibold">
              {t("addModel.classificationMethodConfig.contextBudget.label")}
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
              {t("addModel.classificationMethodConfig.contextBudget.help")}
            </span>
            {contextBudgetQuotesNothing && (
              <span className="block text-xs text-destructive">
                {t("addModel.classificationMethodConfig.contextBudget.tooSmall", {
                  min: MIN_QUOTED_CONTEXT_TURN_CHARS,
                })}
              </span>
            )}
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Switch
                checked={value.classifier_context_include_assistant_turns ?? false}
                onCheckedChange={handleClassifierContextIncludeAssistantTurnsChange}
                size="sm"
                aria-label={t("addModel.classificationMethodConfig.assistantTurns.label")}
              />
              <strong className="font-semibold">{t("addModel.classificationMethodConfig.assistantTurns.label")}</strong>
              <SimpleTooltip content={t("addModel.classificationMethodConfig.assistantTurns.tooltip")}>
                <Info className="size-4 text-muted-foreground" />
              </SimpleTooltip>
            </div>
            <span className="text-xs text-muted-foreground">
              {t("addModel.classificationMethodConfig.assistantTurns.help")}
            </span>
          </div>
        </div>
      )}

      {showHeuristic && heuristic === "heuristic" && (
        <div className="mt-4">
          <div className="flex items-center gap-2 mb-1">
            <strong className="font-semibold">
              {t("addModel.classificationMethodConfig.technicalKeywords.title")}
            </strong>
            <SimpleTooltip content={t("addModel.classificationMethodConfig.technicalKeywords.tooltip")}>
              <Info className="size-4 text-muted-foreground" />
            </SimpleTooltip>
          </div>
          <span className="block mb-2 text-xs text-muted-foreground">
            {t("addModel.classificationMethodConfig.technicalKeywords.help")}
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
            placeholder={t("addModel.classificationMethodConfig.technicalKeywords.placeholder")}
            emptyText={t("addModel.classificationMethodConfig.technicalKeywords.empty")}
            allowCustomValues
            className="w-full"
          />
        </div>
      )}

      {showHeuristic && heuristic === "heuristic" && (
        <>
          {heuristicScoringRole(value) === "decides" && (
            <AutoRouterAllowanceNote
              feature="heuristic_tuning"
              label={t("addModel.classificationMethodConfig.scoringRulesAllowance")}
            />
          )}
          <HeuristicScoringConfig value={value} onChange={onChange} />
          <section className="mt-4 space-y-3">
            <h4 className="font-semibold">{t("addModel.classificationMethodConfig.keywordOverridesTitle")}</h4>
            <HeuristicKeywordOverrides value={value} onChange={onChange} />
          </section>
        </>
      )}
      {showHeuristic && <HowClassificationWorks value={value} />}
    </>
  );
};

export default ClassificationMethodConfig;
