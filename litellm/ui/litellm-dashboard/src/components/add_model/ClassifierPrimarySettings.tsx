import React from "react";
import { useTranslation } from "react-i18next";
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
  section = "all",
}: {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  modelOptions: { value: string; label: string }[];
  showValidationErrors?: boolean;
  section?: "all" | "judge" | "frequency";
}) {
  const { t } = useTranslation();
  const id = React.useId();
  const restriction = restrictedBy(value, "sessionAffinity");
  const frequency = classificationFrequency(value);
  const frequencyDescription = {
    every_request: t("addModel.classifierPrimarySettings.frequency.everyRequestHelp"),
    user_turn: t("addModel.classifierPrimarySettings.frequency.userTurnHelp"),
    session: t("addModel.classifierPrimarySettings.frequency.sessionHelp"),
  }[frequency];
  const usesJudge = usesLlmClassifier(effectiveClassifierType(value));
  const missingJudge = showValidationErrors && usesJudge && !value.classifier_llm_config?.model;
  return (
    <div
      className={
        {
          all: "mb-6 grid gap-4 sm:grid-cols-2",
          frequency: "mb-6 grid gap-4",
          judge: "grid gap-4",
        }[section]
      }
    >
      {section !== "judge" && (
        <div className="space-y-2">
          <Label htmlFor={`${id}-frequency`}>{t("addModel.classifierPrimarySettings.frequency.label")}</Label>
          <Select
            items={[
              { value: "every_request", label: t("addModel.classifierPrimarySettings.frequency.everyRequest") },
              { value: "user_turn", label: t("addModel.classifierPrimarySettings.frequency.userTurn") },
              { value: "session", label: t("addModel.classifierPrimarySettings.frequency.session") },
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
              <SelectItem value="every_request">
                {t("addModel.classifierPrimarySettings.frequency.everyRequest")}
              </SelectItem>
              <SelectItem value="user_turn">{t("addModel.classifierPrimarySettings.frequency.userTurn")}</SelectItem>
              <SelectItem value="session" disabled={Boolean(restriction)}>
                {t("addModel.classifierPrimarySettings.frequency.session")}
              </SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">{restriction?.reason ?? frequencyDescription}</p>
        </div>
      )}
      {section !== "frequency" && usesJudge && (
        <div className="space-y-2">
          <Label htmlFor={`${id}-judge`}>{t("addModel.classifierPrimarySettings.judge.label")}</Label>
          <SearchSelect
            inputId={`${id}-judge`}
            aria-label={t("addModel.classifierPrimarySettings.judge.label")}
            options={modelOptions}
            value={value.classifier_llm_config?.model ?? ""}
            placeholder={t("addModel.classifierPrimarySettings.judge.placeholder")}
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
              {t("addModel.classifierPrimarySettings.judge.required")}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
