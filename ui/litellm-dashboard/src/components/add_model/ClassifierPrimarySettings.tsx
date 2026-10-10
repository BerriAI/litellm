import React from "react";
import { Label } from "@/components/ui/label";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  classificationFrequency,
  withClassificationFrequency,
  effectiveClassifierType,
  usesLlmClassifier,
  DEFAULT_CLASSIFIER_TIMEOUT_MS,
  type ComplexityRouterConfigValue,
  type ClassificationFrequency,
} from "./ComplexityRouterConfig";
import { restrictedBy } from "./TierRestrictions";

export default function ClassifierPrimarySettings({
  value,
  onChange,
  modelOptions,
  showValidationErrors = false,
}: {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  modelOptions: { value: string; label: string }[];
  showValidationErrors?: boolean;
}) {
  const id = React.useId();
  const restriction = restrictedBy(value, "sessionAffinity");
  const frequency = classificationFrequency(value);
  const frequencyDescription = {
    every_request: "Choose a model again for every request",
    user_turn: "Reclassify when the user sends a new message",
    session: "Keep the same tier for the session. Requires a client session ID",
  }[frequency];
  const usesJudge = usesLlmClassifier(effectiveClassifierType(value));
  const missingJudge = showValidationErrors && usesJudge && !value.classifier_llm_config?.model;
  return (
    <div className="mb-6 grid gap-4 sm:grid-cols-2">
      <div className="space-y-2">
        <Label htmlFor={`${id}-frequency`}>How often to classify</Label>
        <Select
          items={[
            { value: "every_request", label: "Every request" },
            { value: "user_turn", label: "Every new user message" },
            { value: "session", label: "Once per session" },
          ]}
          value={frequency}
          onValueChange={(frequency) => {
            if (frequency) onChange(withClassificationFrequency(value, frequency as ClassificationFrequency));
          }}
        >
          <SelectTrigger id={`${id}-frequency`} className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="every_request">Every request</SelectItem>
            <SelectItem value="user_turn">Every new user message</SelectItem>
            <SelectItem value="session" disabled={Boolean(restriction)}>
              Once per session
            </SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">{restriction?.reason ?? frequencyDescription}</p>
      </div>
      {usesJudge && (
        <div className="space-y-2">
          <Label htmlFor={`${id}-judge`}>Judge model</Label>
          <SearchSelect
            inputId={`${id}-judge`}
            aria-label="Judge model"
            options={modelOptions}
            value={value.classifier_llm_config?.model ?? ""}
            placeholder="Select the judge model"
            allowClear={false}
            className={missingJudge ? "border-destructive" : undefined}
            onValueChange={(model) => {
              if (!model || model === value.classifier_llm_config?.model) return;
              onChange({
                ...value,
                classifier_llm_config: {
                  ...value.classifier_llm_config,
                  model,
                  timeout_ms: value.classifier_llm_config?.timeout_ms ?? DEFAULT_CLASSIFIER_TIMEOUT_MS,
                  reasoning_effort: undefined,
                },
              });
            }}
          />
          {missingJudge && (
            <p role="alert" className="text-xs text-destructive">
              A judge model is required
            </p>
          )}
        </div>
      )}
    </div>
  );
}
