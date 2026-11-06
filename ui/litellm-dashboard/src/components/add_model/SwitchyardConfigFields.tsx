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

interface Props {
  value: SwitchyardConfig;
  onChange: (value: SwitchyardConfig) => void;
  models: ModelChoice[];
  decisionModels: ModelChoice[];
  onToolSemanticsError: (error: string | null) => void;
}

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
            Heuristic
          </span>
          <p
            aria-labelledby={`${id}-heuristic`}
            className="flex min-h-10 items-center rounded-md border bg-muted/30 px-3 py-2 text-sm"
          >
            Stage Router
          </p>
        </div>
        <div className="min-w-0 space-y-2">
          <label htmlFor={`${id}-bias`} className="text-sm font-medium">
            Bias
          </label>
          <Select
            value={value.options.picker}
            items={[
              { value: "efficient_first", label: "Cost-first" },
              { value: "capable_first", label: "Quality-first (experimental)" },
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
              <SelectItem value="efficient_first">Cost-first</SelectItem>
              <SelectItem value="capable_first">
                <span className="whitespace-normal">Quality-first (experimental)</span>
              </SelectItem>
            </SelectContent>
          </Select>
          {value.options.picker === "capable_first" && (
            <p className="text-sm text-muted-foreground">
              Switchyard has not benchmarked this bias, so it has no calibrated confidence thresholds.
            </p>
          )}
        </div>
        {(["capable", "efficient"] as const).map((tier) => (
          <div key={tier} className="min-w-0 space-y-2">
            <label htmlFor={`${id}-${tier}`} className="text-sm font-medium">
              {tier === "capable" ? "Capable model" : "Efficient model"}
            </label>
            <ModelChoiceCombobox
              id={`${id}-${tier}`}
              value={value.models[tier]}
              onChange={(model) => onChange({ ...value, models: { ...value.models, [tier]: model ?? "" } })}
              choices={models}
              placeholder="Select a model"
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
              Classifier
            </label>
            <p className="text-sm text-muted-foreground">
              Ask a judge model when the tool history does not give a confident routing decision.
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
                Classifier type
              </label>
              <Select
                value={classifier.kind}
                items={[
                  { value: "llm", label: "LLM judge" },
                  { value: "decision", label: "Decision model" },
                ]}
                onValueChange={(kind) => onChange(setClassifierKind(value, kind === "decision" ? "decision" : "llm"))}
              >
                <SelectTrigger id={`${id}-classifier-kind`} className="min-h-10 w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="llm">LLM judge</SelectItem>
                  <SelectItem value="decision">Decision model</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="min-w-0 space-y-2">
              <label htmlFor={`${id}-judge`} className="text-sm font-medium">
                {decision ? "Decision model" : "Judge model"}
              </label>
              <ModelChoiceCombobox
                id={`${id}-judge`}
                value={value.models.judge ?? ""}
                onChange={(judge) => onChange({ ...value, models: { ...value.models, judge: judge ?? "" } })}
                choices={decision ? decisionModels : models}
                placeholder={decision ? "Select a decision model" : "Select a judge model"}
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
            Confidence threshold
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
          Below this confidence, use {classifier ? "the judge model" : "the bias’s default model"}. Critical errors can
          still select the capable model.
        </p>
      </div>
      <details className="rounded-lg border p-4">
        <summary className="cursor-pointer text-sm font-medium">Advanced settings</summary>
        <div className="mt-5 space-y-5">
          <h3 className="text-sm font-semibold">Stage</h3>
          <div className="min-w-0 space-y-2">
            <label htmlFor={`${id}-recent`}>Recent window</label>
            <Input
              id={`${id}-recent`}
              type="number"
              min={1}
              value={value.options.recent_window ?? ""}
              placeholder="Library default"
              onChange={(event) => setOption("recent_window", event.target.value)}
            />
          </div>
          <div className="min-w-0 space-y-2">
            <label htmlFor={`${id}-tools`}>Tool semantics</label>
            <Textarea
              id={`${id}-tools`}
              value={toolSemantics}
              placeholder={'{"observe": ["read_file"], "mutate": ["edit_file"]}'}
              onChange={(event) => changeToolSemantics(event.target.value)}
            />
          </div>
          {decision && (
            <>
              <h3 className="text-sm font-semibold">Decision model</h3>
              <div className="min-w-0 space-y-2">
                <label htmlFor={`${id}-cutoff`}>Advantage cutoff</label>
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
                  Route to the capable model when the decision model’s advantage probability is above this value.
                </p>
              </div>
              <div className="min-w-0 space-y-2">
                <label htmlFor={`${id}-instructions`}>Decision instructions</label>
                <Textarea
                  id={`${id}-instructions`}
                  readOnly={isStructured(decision.instructions)}
                  value={instructionsText(decision.instructions)}
                  placeholder="Use Switchyard’s built-in instructions"
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
              <h3 className="text-sm font-semibold">Classifier</h3>
              <div className="grid gap-4 md:grid-cols-2">
                {(
                  [
                    ["base_threshold", "Capability threshold"],
                    ["threshold_step", "Threshold step"],
                    ["max_output_tokens", "Maximum output tokens"],
                    ["recent_turn_window", "Recent turn window"],
                  ] as const
                ).map(([key, label]) => (
                  <div key={key} className="min-w-0 space-y-2">
                    <label htmlFor={`${id}-${key}`}>{label}</label>
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
                <label htmlFor={`${id}-format`}>Response format</label>
                <Select
                  value={llm.response_format_type}
                  items={[
                    { value: "json_schema", label: "JSON schema" },
                    { value: "json_object", label: "JSON object" },
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
                    <SelectItem value="json_schema">JSON schema</SelectItem>
                    <SelectItem value="json_object">JSON object</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="min-w-0 space-y-2">
                <label htmlFor={`${id}-prompt`}>Classifier prompt</label>
                <Textarea
                  id={`${id}-prompt`}
                  value={llm.prompt ?? ""}
                  placeholder="Use Switchyard’s built-in prompt"
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
