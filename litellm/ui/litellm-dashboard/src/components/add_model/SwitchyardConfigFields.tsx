"use client";

import { useId, useState } from "react";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import ModelChoiceCombobox, { type ModelChoice } from "./ModelChoiceCombobox";
import {
  stageConfidence,
  toggleClassifier,
  setClassifierKind,
  parseToolSemantics,
  type SwitchyardConfig,
} from "./switchyard_config";
import { useTranslation } from "react-i18next";
import { codeSamples } from "@/i18n/untranslated/codeSamples";

interface Props {
  value: SwitchyardConfig;
  onChange: (value: SwitchyardConfig) => void;
  models: ModelChoice[];
  decisionModels: ModelChoice[];
  onToolSemanticsError: (error: string | null) => void;
}

const TIERS = ["capable", "efficient"] as const;

const LLM_NUMBER_FIELDS = [
  ["base_threshold", "addModel.switchyardConfigFields.llmFields.capabilityThreshold"],
  ["threshold_step", "addModel.switchyardConfigFields.llmFields.thresholdStep"],
  ["max_output_tokens", "addModel.switchyardConfigFields.llmFields.maxOutputTokens"],
  ["recent_turn_window", "addModel.switchyardConfigFields.llmFields.recentTurnWindow"],
] as const;

const isStructured = (instructions: unknown): instructions is Record<string, unknown> =>
  typeof instructions === "object" && instructions !== null;
const instructionsText = (instructions: string | Record<string, unknown> | null | undefined): string =>
  isStructured(instructions) ? JSON.stringify(instructions, null, 2) : instructions ?? "";

export default function SwitchyardConfigFields({
  value,
  onChange,
  models,
  decisionModels,
  onToolSemanticsError,
}: Props) {
  const { t } = useTranslation();
  const id = useId();
  const [toolSemantics, setToolSemantics] = useState(
    value.options.tool_semantics ? JSON.stringify(value.options.tool_semantics, null, 2) : "",
  );
  const classifier = value.classifier;
  const llm = classifier?.kind === "llm" ? classifier : undefined;
  const decision = classifier?.kind === "decision" ? classifier : undefined;
  const setOption = (key: "recent_window", text: string) =>
    onChange({ ...value, options: { ...value.options, [key]: text === "" ? undefined : Number(text) } });
  const setClassifierNumber = (
    key: "base_threshold" | "threshold_step" | "max_output_tokens" | "recent_turn_window",
    text: string,
  ) => {
    if (llm) onChange({ ...value, classifier: { ...llm, [key]: text === "" ? undefined : Number(text) } });
  };
  const changeToolSemantics = (text: string) => {
    setToolSemantics(text);
    try {
      const parsed = parseToolSemantics(text);
      onChange({ ...value, options: { ...value.options, tool_semantics: parsed } });
      onToolSemanticsError(null);
    } catch {
      onToolSemanticsError("Tool semantics must be JSON mapping observe, mutate, plan, or new to arrays of tool names");
    }
  };
  return (
    <div className="space-y-6">
      <div className="grid gap-5 md:grid-cols-2">
        <div className="min-w-0 space-y-2">
          <span id={`${id}-heuristic`} className="text-sm font-medium">
            {t("addModel.switchyardConfigFields.heuristic")}
          </span>
          <p
            aria-labelledby={`${id}-heuristic`}
            className="flex min-h-10 items-center rounded-md border bg-muted/30 px-3 py-2 text-sm"
          >
            {t("addModel.switchyardConfigFields.stageRouter")}
          </p>
        </div>
        <div className="min-w-0 space-y-2">
          <label htmlFor={`${id}-bias`} className="text-sm font-medium">
            {t("addModel.switchyardConfigFields.bias")}
          </label>
          <Select
            value={value.options.picker}
            items={[
              { value: "efficient_first", label: t("addModel.switchyardConfigFields.biasOptions.costFirst") },
              { value: "capable_first", label: t("addModel.switchyardConfigFields.biasOptions.qualityFirst") },
            ]}
            onValueChange={(picker) =>
              onChange({
                ...value,
                options: { ...value.options, picker: picker === "capable_first" ? "capable_first" : "efficient_first" },
              })
            }
          >
            <SelectTrigger
              id={`${id}-bias`}
              className="h-auto min-h-10 w-full py-2 *:data-[slot=select-value]:line-clamp-none"
            >
              <SelectValue className="whitespace-normal break-words" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="efficient_first">
                {t("addModel.switchyardConfigFields.biasOptions.costFirst")}
              </SelectItem>
              <SelectItem value="capable_first">
                <span className="whitespace-normal">
                  {t("addModel.switchyardConfigFields.biasOptions.qualityFirst")}
                </span>
              </SelectItem>
            </SelectContent>
          </Select>
          {value.options.picker === "capable_first" && (
            <p className="text-sm text-muted-foreground">{t("addModel.switchyardConfigFields.qualityFirstWarning")}</p>
          )}
        </div>
        {TIERS.map((tier) => (
          <div key={tier} className="min-w-0 space-y-2">
            <label htmlFor={`${id}-${tier}`} className="text-sm font-medium">
              {tier === "capable"
                ? t("addModel.switchyardConfigFields.capableModel")
                : t("addModel.switchyardConfigFields.efficientModel")}
            </label>
            <ModelChoiceCombobox
              id={`${id}-${tier}`}
              value={value.models[tier]}
              onChange={(model) => onChange({ ...value, models: { ...value.models, [tier]: model ?? "" } })}
              choices={models}
              placeholder={t("addModel.switchyardConfigFields.selectModel")}
              ariaInvalid={undefined}
              ariaDescribedBy={undefined}
            />
          </div>
        ))}
      </div>
      <div className="rounded-lg border p-4 space-y-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <label htmlFor={`${id}-classifier`} className="text-sm font-medium">
              {t("addModel.switchyardConfigFields.classifier")}
            </label>
            <p className="text-sm text-muted-foreground">
              {t("addModel.switchyardConfigFields.classifierDescription")}
            </p>
          </div>
          <Switch
            id={`${id}-classifier`}
            checked={Boolean(classifier)}
            onCheckedChange={(enabled) => onChange(toggleClassifier(value, enabled))}
          />
        </div>
        {classifier && (
          <div className="grid gap-5 md:grid-cols-2">
            <div className="min-w-0 space-y-2">
              <label htmlFor={`${id}-classifier-kind`} className="text-sm font-medium">
                {t("addModel.switchyardConfigFields.classifierType")}
              </label>
              <Select
                value={classifier.kind}
                items={[
                  { value: "llm", label: t("addModel.switchyardConfigFields.classifierKinds.llm") },
                  { value: "decision", label: t("addModel.switchyardConfigFields.classifierKinds.decision") },
                ]}
                onValueChange={(kind) => onChange(setClassifierKind(value, kind === "decision" ? "decision" : "llm"))}
              >
                <SelectTrigger id={`${id}-classifier-kind`} className="min-h-10 w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="llm">{t("addModel.switchyardConfigFields.classifierKinds.llm")}</SelectItem>
                  <SelectItem value="decision">
                    {t("addModel.switchyardConfigFields.classifierKinds.decision")}
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="min-w-0 space-y-2">
              <label htmlFor={`${id}-judge`} className="text-sm font-medium">
                {decision
                  ? t("addModel.switchyardConfigFields.decisionModel")
                  : t("addModel.switchyardConfigFields.judgeModel")}
              </label>
              <ModelChoiceCombobox
                id={`${id}-judge`}
                value={value.models.judge ?? ""}
                onChange={(judge) => onChange({ ...value, models: { ...value.models, judge: judge ?? "" } })}
                choices={decision ? decisionModels : models}
                placeholder={
                  decision
                    ? t("addModel.switchyardConfigFields.selectDecisionModel")
                    : t("addModel.switchyardConfigFields.selectJudgeModel")
                }
                ariaInvalid={undefined}
                ariaDescribedBy={undefined}
              />
            </div>
          </div>
        )}
      </div>
      <div className="space-y-3">
        <div className="flex justify-between">
          <span id={`${id}-confidence`} className="text-sm font-medium">
            {t("addModel.switchyardConfigFields.confidenceThreshold")}
          </span>
          <span className="text-sm tabular-nums">{stageConfidence(value).toFixed(2)}</span>
        </div>
        <Slider
          aria-labelledby={`${id}-confidence`}
          value={[stageConfidence(value)]}
          min={0}
          max={1}
          step={0.01}
          onValueChange={(next) =>
            onChange({
              ...value,
              options: { ...value.options, confidence_threshold: Array.isArray(next) ? next[0] : next },
            })
          }
        />
        <p className="text-sm text-muted-foreground">
          {classifier
            ? t("addModel.switchyardConfigFields.fallbackToJudge")
            : t("addModel.switchyardConfigFields.fallbackToDefault")}
        </p>
      </div>
      <details className="rounded-lg border p-4">
        <summary className="cursor-pointer text-sm font-medium">
          {t("addModel.switchyardConfigFields.advancedSettings")}
        </summary>
        <div className="mt-5 space-y-5">
          <h3 className="text-sm font-semibold">{t("addModel.switchyardConfigFields.sections.stage")}</h3>
          <div className="min-w-0 space-y-2">
            <label htmlFor={`${id}-recent`}>{t("addModel.switchyardConfigFields.recentWindow")}</label>
            <Input
              id={`${id}-recent`}
              type="number"
              min={1}
              value={value.options.recent_window ?? ""}
              placeholder={t("addModel.switchyardConfigFields.libraryDefault")}
              onChange={(event) => setOption("recent_window", event.target.value)}
            />
          </div>
          <div className="min-w-0 space-y-2">
            <label htmlFor={`${id}-tools`}>{t("addModel.switchyardConfigFields.toolSemantics")}</label>
            <Textarea
              id={`${id}-tools`}
              value={toolSemantics}
              placeholder={codeSamples.models.toolSemanticsJson}
              onChange={(event) => changeToolSemantics(event.target.value)}
            />
          </div>
          {decision && (
            <>
              <h3 className="text-sm font-semibold">{t("addModel.switchyardConfigFields.sections.decisionModel")}</h3>
              <div className="min-w-0 space-y-2">
                <label htmlFor={`${id}-cutoff`}>{t("addModel.switchyardConfigFields.advantageCutoff")}</label>
                <Input
                  id={`${id}-cutoff`}
                  type="number"
                  min={0}
                  max={1}
                  step={0.01}
                  value={decision.cutoff}
                  onChange={(event) =>
                    onChange({ ...value, classifier: { ...decision, cutoff: Number(event.target.value) } })
                  }
                />
                <p className="text-sm text-muted-foreground">
                  {t("addModel.switchyardConfigFields.advantageCutoffHelp")}
                </p>
              </div>
              <div className="min-w-0 space-y-2">
                <label htmlFor={`${id}-instructions`}>
                  {t("addModel.switchyardConfigFields.decisionInstructions")}
                </label>
                <Textarea
                  id={`${id}-instructions`}
                  readOnly={isStructured(decision.instructions)}
                  value={instructionsText(decision.instructions)}
                  placeholder={t("addModel.switchyardConfigFields.decisionInstructionsPlaceholder")}
                  onChange={(event) =>
                    onChange({
                      ...value,
                      classifier: { ...decision, instructions: event.target.value || undefined },
                    })
                  }
                />
              </div>
            </>
          )}
          {llm && (
            <>
              <h3 className="text-sm font-semibold">{t("addModel.switchyardConfigFields.sections.classifier")}</h3>
              <div className="grid gap-4 md:grid-cols-2">
                {LLM_NUMBER_FIELDS.map(([key, labelKey]) => (
                  <div key={key} className="min-w-0 space-y-2">
                    <label htmlFor={`${id}-${key}`}>{t(labelKey)}</label>
                    <Input
                      id={`${id}-${key}`}
                      type="number"
                      min={key.includes("threshold") ? 0 : 1}
                      max={key.includes("threshold") ? 1 : undefined}
                      step={key.includes("threshold") ? 0.01 : 1}
                      value={llm[key] ?? ""}
                      onChange={(event) => setClassifierNumber(key, event.target.value)}
                    />
                  </div>
                ))}
              </div>
              <div className="min-w-0 space-y-2">
                <label htmlFor={`${id}-format`}>{t("addModel.switchyardConfigFields.responseFormat")}</label>
                <Select
                  value={llm.response_format_type}
                  items={[
                    { value: "json_schema", label: t("addModel.switchyardConfigFields.responseFormats.jsonSchema") },
                    { value: "json_object", label: t("addModel.switchyardConfigFields.responseFormats.jsonObject") },
                  ]}
                  onValueChange={(format) =>
                    onChange({
                      ...value,
                      classifier: {
                        ...llm,
                        response_format_type: format === "json_object" ? "json_object" : "json_schema",
                      },
                    })
                  }
                >
                  <SelectTrigger id={`${id}-format`} className="min-h-10 w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="json_schema">
                      {t("addModel.switchyardConfigFields.responseFormats.jsonSchema")}
                    </SelectItem>
                    <SelectItem value="json_object">
                      {t("addModel.switchyardConfigFields.responseFormats.jsonObject")}
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="min-w-0 space-y-2">
                <label htmlFor={`${id}-prompt`}>{t("addModel.switchyardConfigFields.classifierPrompt")}</label>
                <Textarea
                  id={`${id}-prompt`}
                  value={llm.prompt ?? ""}
                  placeholder={t("addModel.switchyardConfigFields.classifierPromptPlaceholder")}
                  onChange={(event) =>
                    onChange({ ...value, classifier: { ...llm, prompt: event.target.value || undefined } })
                  }
                />
              </div>
            </>
          )}
        </div>
      </details>
    </div>
  );
}
