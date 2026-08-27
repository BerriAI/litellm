import { Info } from "lucide-react";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Switch } from "@/components/ui/switch";
import React from "react";
import ClassifierPromptEditor from "./ClassifierPromptEditor";
import HeuristicScoringConfig from "./HeuristicScoringConfig";
import { useComplexityScorerDefaults } from "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults";
import {
  ClassifierFallback,
  ClassifierType,
  ComplexityRouterConfigValue,
  DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS,
  MIN_QUOTED_CONTEXT_TURN_CHARS,
  DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE,
  DEFAULT_CLASSIFIER_FALLBACK,
  DEFAULT_CLASSIFIER_TIMEOUT_MS,
  DEFAULT_CLASSIFICATION_RUBRIC,
  NEW_CLASSIFIER_CLASSIFICATION_RUBRIC,
  CLASSIFICATION_RUBRIC_DESCRIPTIONS,
  CLASSIFICATION_RUBRIC_KEYS,
  ClassificationRubric,
  effectiveTierLabel,
  heuristicScoringRole,
  usesLlmClassifier,
  DEFAULT_HEURISTIC_FIRST_MAX_TIER,
  HEURISTIC_FIRST_MAX_TIER_KEYS,
} from "./ComplexityRouterConfig";

const DEFAULT_SCORING_EXPLANATION =
  "The router scores each request across 7 dimensions: token count, code presence, reasoning markers, technical " +
  "terms, simple indicators, multi-step patterns, and question complexity. The weighted score determines the tier:";

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
  const ranges = boundaryRanges(
    scorerDefaults?.tier_boundaries,
    value.tier_boundaries,
    value.reasoning_override_min_score,
  );

  return (
    <Card className="bg-muted mt-4">
      <CardContent>
        <strong className="block mb-2 font-semibold">How Classification Works</strong>
        <span className="text-[13px] text-muted-foreground">{scoringExplanation(value)}</span>
        {ranges && (
          <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20, fontSize: 13, color: "rgba(0, 0, 0, 0.45)" }}>
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
  customTechnicalKeywords?: string[];
  onCustomTechnicalKeywordsChange?: (keywords: string[]) => void;
  showValidationErrors?: boolean;
  /** The resolved default model - see resolveComplexityDefaultModel. Names and gates the radio. */
  defaultModel?: string;
}

const ClassificationMethodConfig: React.FC<ClassificationMethodConfigProps> = ({
  value,
  onChange,
  modelOptions,
  customTechnicalKeywords,
  onCustomTechnicalKeywordsChange,
  showValidationErrors = false,
  defaultModel,
}) => {
  const hasDefaultModel = Boolean(defaultModel);
  const classifierModelMissing =
    showValidationErrors && usesLlmClassifier(value.classifier_type) && !value.classifier_llm_config?.model;
  const usesCustomPrompt = Boolean(value.classifier_llm_config?.system_prompt?.trim());
  const contextBudget = value.classifier_context_budget_chars ?? DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS;
  const contextBudgetQuotesNothing = contextBudget > 0 && contextBudget < MIN_QUOTED_CONTEXT_TURN_CHARS;
  const classificationRubric = value.classifier_llm_config?.classification_rubric ?? DEFAULT_CLASSIFICATION_RUBRIC;

  const handleClassifierTypeChange = (classifierType: ClassifierType) => {
    const nextValue: ComplexityRouterConfigValue = {
      ...value,
      classifier_type: classifierType,
      classifier_llm_config: usesLlmClassifier(classifierType)
        ? value.classifier_llm_config ?? {
            model: "",
            timeout_ms: DEFAULT_CLASSIFIER_TIMEOUT_MS,
            classification_rubric: NEW_CLASSIFIER_CLASSIFICATION_RUBRIC,
          }
        : undefined,
      classifier_context_window_size: usesLlmClassifier(classifierType)
        ? value.classifier_context_window_size ?? DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE
        : undefined,
      classifier_context_budget_chars: usesLlmClassifier(classifierType)
        ? value.classifier_context_budget_chars ?? DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS
        : undefined,
      classifier_context_include_assistant_turns: usesLlmClassifier(classifierType)
        ? value.classifier_context_include_assistant_turns
        : undefined,
      classifier_fallback: usesLlmClassifier(classifierType) ? value.classifier_fallback : undefined,
      heuristic_first_max_tier:
        classifierType === "heuristic_first"
          ? value.heuristic_first_max_tier ?? DEFAULT_HEURISTIC_FIRST_MAX_TIER
          : undefined,
    };
    onChange(nextValue);
  };

  const handleHeuristicFirstMaxTierChange = (tier: string) => {
    onChange({ ...value, heuristic_first_max_tier: tier });
  };

  const handleClassifierModelChange = (model: string) => {
    onChange({
      ...value,
      classifier_llm_config: {
        ...value.classifier_llm_config,
        model,
        timeout_ms: value.classifier_llm_config?.timeout_ms ?? DEFAULT_CLASSIFIER_TIMEOUT_MS,
      },
    });
  };

  const handleClassifierTimeoutChange = (timeoutMs: number | null) => {
    onChange({
      ...value,
      classifier_llm_config: {
        ...value.classifier_llm_config,
        model: value.classifier_llm_config?.model ?? "",
        timeout_ms: timeoutMs ?? DEFAULT_CLASSIFIER_TIMEOUT_MS,
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

  const handleClassifierContextWindowSizeChange = (windowSize: number | null) => {
    onChange({
      ...value,
      classifier_context_window_size: windowSize ?? DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE,
    });
  };

  const handleClassifierContextBudgetCharsChange = (budgetChars: number | null) => {
    onChange({
      ...value,
      classifier_context_budget_chars: budgetChars ?? DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS,
    });
  };

  const handleClassifierContextIncludeAssistantTurnsChange = (includeAssistantTurns: boolean) => {
    onChange({
      ...value,
      classifier_context_include_assistant_turns: includeAssistantTurns,
    });
  };

  return (
    <>
      <RadioGroup
        value={value.classifier_type}
        onValueChange={(classifierType: unknown) => handleClassifierTypeChange(classifierType as ClassifierType)}
        className="w-full"
      >
        <div className="flex w-full flex-col items-start gap-2">
          <Label className="items-start font-normal leading-normal">
            <RadioGroupItem value="heuristic" className="mt-0.5" />
            <span>
              <strong className="font-semibold">Heuristic</strong>{" "}
              <span className="text-muted-foreground">
                (default), rule-based scoring with no API calls and &lt;1ms latency
              </span>
            </span>
          </Label>
          <Label className="items-start font-normal leading-normal">
            <RadioGroupItem value="llm" className="mt-0.5" />
            <span>
              <strong className="font-semibold">LLM Classifier</strong>{" "}
              <span className="text-muted-foreground">calls a model to decide the tier (e.g. a small/fast model)</span>
            </span>
          </Label>
          <Label className="items-start font-normal leading-normal">
            <RadioGroupItem value="heuristic_first" className="mt-0.5" />
            <span>
              <strong className="font-semibold">Heuristic first</strong>{" "}
              <span className="text-muted-foreground">
                scores locally, and only pays for the classifier when the score does not confidently land a cheap tier
              </span>
            </span>
          </Label>
        </div>
      </RadioGroup>

      {value.classifier_type === "heuristic_first" && (
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
        </div>
      )}

      {usesLlmClassifier(value.classifier_type) && (
        <div className="mt-4 space-y-3">
          <div>
            <strong className="block mb-1 font-semibold">Classifier Model</strong>
            <SearchSelect
              options={modelOptions}
              value={value.classifier_llm_config?.model ?? ""}
              onValueChange={handleClassifierModelChange}
              placeholder="Select the model that will classify request complexity"
              emptyText="No models found"
              allowClear={false}
              className={classifierModelMissing ? "border-destructive" : undefined}
            />
            {classifierModelMissing && <span className="text-xs text-destructive">A classifier model is required</span>}
          </div>
          <div>
            <strong className="block mb-1 font-semibold">Timeout (ms)</strong>
            <Input
              type="number"
              value={value.classifier_llm_config?.timeout_ms ?? DEFAULT_CLASSIFIER_TIMEOUT_MS}
              onChange={(event) =>
                handleClassifierTimeoutChange(event.target.value === "" ? null : event.target.valueAsNumber)
              }
              min={1}
              className="w-full"
            />
            <span className="text-xs text-muted-foreground">
              How long the classifier call has before it fails and the fallback below takes over.
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <strong className="font-semibold">Classification Rubric</strong>
              <SimpleTooltip content="Every rubric uses the same four tiers. They differ in the worked examples that show the classifier where the boundary between tiers sits, and the Business rubric also rewrites the tier definitions for business traffic.">
                <Info className="size-4 text-muted-foreground" />
              </SimpleTooltip>
            </div>
            <SimpleTooltip
              content={usesCustomPrompt ? "Your custom prompt replaces the built-in rubric entirely" : undefined}
              className="w-full"
            >
              <Select
                items={CLASSIFICATION_RUBRIC_KEYS.map((preset) => ({
                  value: preset,
                  label: CLASSIFICATION_RUBRIC_DESCRIPTIONS[preset].label,
                }))}
                value={classificationRubric}
                onValueChange={(preset: ClassificationRubric | null) =>
                  preset && handleClassificationRubricChange(preset)
                }
                disabled={usesCustomPrompt}
              >
                <SelectTrigger aria-label="Classification Rubric" className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {CLASSIFICATION_RUBRIC_KEYS.map((preset) => (
                    <SelectItem key={preset} value={preset}>
                      {CLASSIFICATION_RUBRIC_DESCRIPTIONS[preset].label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </SimpleTooltip>
            <span className="block text-xs text-muted-foreground">
              {usesCustomPrompt
                ? "Not in use: the custom prompt below is the classifier's entire rubric."
                : CLASSIFICATION_RUBRIC_DESCRIPTIONS[classificationRubric].description}
            </span>
          </div>
          <div>
            <strong className="block mb-1 font-semibold">Classifier Prompt</strong>
            <ClassifierPromptEditor
              systemPrompt={value.classifier_llm_config?.system_prompt}
              onChange={handleClassifierSystemPromptChange}
              contextWindowSize={value.classifier_context_window_size ?? DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE}
              tierLabels={value.tier_labels}
              classificationRubric={classificationRubric}
            />
          </div>
          <div>
            <strong className="block mb-1 font-semibold">If the classifier fails</strong>
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
          </div>
          <div>
            <strong className="block mb-1 font-semibold">Context Window Size</strong>
            <Input
              type="number"
              value={value.classifier_context_window_size ?? DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE}
              onChange={(event) =>
                handleClassifierContextWindowSizeChange(event.target.value === "" ? null : event.target.valueAsNumber)
              }
              min={0}
              className="w-full"
            />
            <span className="text-xs text-muted-foreground">
              Number of prior user turns (tool output and harness reminders excluded) sent to the classifier as context,
              so a referring follow-up like &quot;now do the same for the streaming path&quot; is classified against
              what it refers to. Set to 0 to send only the current message.
            </span>
          </div>
          <div>
            <strong className="block mb-1 font-semibold">Context Character Budget</strong>
            <Input
              type="number"
              value={value.classifier_context_budget_chars ?? DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS}
              onChange={(event) =>
                handleClassifierContextBudgetCharsChange(event.target.value === "" ? null : event.target.valueAsNumber)
              }
              min={0}
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

      <HeuristicScoringConfig value={value} onChange={onChange} />

      <HowClassificationWorks value={value} />
    </>
  );
};

export default ClassificationMethodConfig;
