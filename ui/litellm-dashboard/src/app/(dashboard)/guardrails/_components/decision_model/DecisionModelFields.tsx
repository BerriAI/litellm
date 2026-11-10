"use client";

import type { ParseKeys } from "i18next";
import { Info, Plus, X } from "lucide-react";
import React from "react";
import { useWatch } from "react-hook-form";
import { Trans, useTranslation } from "react-i18next";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Field, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Logo } from "@/components/molecules/logo/Logo";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { uiHref } from "@/utils/uiHref";
import { asText, GuardrailField, labelWithHint, requiredRule, type GuardrailFormControl } from "../GuardrailFormField";
import { duplicateDecisionCheckNames, type DecisionModelCheckDraft } from "./buildDecisionModelParams";
import DecisionTestSection from "./DecisionTestSection";
import { newDecisionCheckDraft, type DecisionModelOption } from "./decisionModelQuestion";

export interface DecisionModelFieldsProps {
  accessToken: string | null;
  providerFilter: string | null;
  onModelChange: () => void;
  decisionModels: DecisionModelOption[];
  checks: DecisionModelCheckDraft[];
  onChecksChange: (checks: DecisionModelCheckDraft[]) => void;
  control: GuardrailFormControl;
}

const ACTION_ITEMS: { labelKey: ParseKeys; value: "block" | "log" }[] = [
  { labelKey: "guardrails.decisionModel.actionOptions.block", value: "block" },
  { labelKey: "guardrails.decisionModel.actionOptions.log", value: "log" },
];

const ProviderLogo: React.FC<{ provider: string }> = ({ provider }) => (
  <Logo provider={provider} label={getProviderLogoAndName(provider).displayName} className="size-4 shrink-0" />
);

const NoDecisionModels: React.FC<{ providerFilter: string | null }> = ({ providerFilter }) => {
  const { t } = useTranslation();
  const description = providerFilter
    ? t("guardrails.decisionModel.noModels.descriptionForProvider", {
        provider: getProviderLogoAndName(providerFilter).displayName,
      })
    : t("guardrails.decisionModel.noModels.description");
  return (
    <Alert variant="info" role="note" aria-label={t("guardrails.decisionModel.noModels.ariaLabel")}>
      <Info />
      <AlertTitle>{t("guardrails.decisionModel.noModels.title")}</AlertTitle>
      <AlertDescription>
        <p>
          {description}{" "}
          <a href={uiHref("models-and-endpoints")} target="_blank" rel="noopener noreferrer">
            {t("guardrails.decisionModel.noModels.addModel")}
          </a>
        </p>
      </AlertDescription>
    </Alert>
  );
};

const ThresholdSlider: React.FC<{
  value: number;
  onChange: (value: number) => void;
  "aria-label": string;
}> = ({ value, onChange, "aria-label": ariaLabel }) => (
  <div className="flex items-center gap-2">
    <Slider
      aria-label={ariaLabel}
      min={0}
      max={1}
      step={0.01}
      value={[value]}
      onValueChange={(next) => onChange(Array.isArray(next) ? next[0] ?? 0 : next)}
      className="min-w-[120px] flex-1 **:data-[slot=slider-track]:bg-muted-foreground/25"
    />
    <span className="w-9 whitespace-nowrap text-right text-xs tabular-nums text-muted-foreground">
      {value.toFixed(2)}
    </span>
  </div>
);

const DecisionModelFields: React.FC<DecisionModelFieldsProps> = ({
  accessToken,
  providerFilter,
  onModelChange,
  decisionModels,
  checks,
  onChecksChange,
  control,
}) => {
  const { t } = useTranslation();
  const selectedModel = asText(useWatch({ control, name: "decision_model" }));
  const duplicateNames = duplicateDecisionCheckNames(checks);
  const providerByModel = new Map(decisionModels.map((option) => [option.model, option.provider]));
  const actionItems = ACTION_ITEMS.map(({ labelKey, value }) => ({ label: t(labelKey), value }));

  const updateCheck = (id: string, patch: Partial<DecisionModelCheckDraft>) =>
    onChecksChange(checks.map((check) => (check.id === id ? { ...check, ...patch } : check)));

  const addCheck = () => onChecksChange([...checks, newDecisionCheckDraft(checks)]);

  const removeCheck = (id: string) => onChecksChange(checks.filter((check) => check.id !== id));

  return (
    <div className="space-y-5">
      <div className="rounded-md border border-success/20 bg-success/10 px-3.5 py-2.5 text-[13px] text-success">
        <Trans i18nKey="guardrails.decisionModel.intro" components={{ strong: <strong /> }} />
      </div>

      <GuardrailField
        control={control}
        name="decision_model"
        label={labelWithHint(t("guardrails.decisionModel.model.label"), t("guardrails.decisionModel.model.hint"))}
        rules={requiredRule(t("guardrails.decisionModel.model.required"))}
      >
        {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
          <Combobox
            items={decisionModels.map((option) => option.model)}
            value={asText(value) || null}
            onValueChange={(next: string | null) => {
              onChange(next);
              onModelChange();
            }}
          >
            <ComboboxInput
              id={id}
              aria-invalid={ariaInvalid}
              aria-describedby={ariaDescribedBy}
              placeholder={
                decisionModels.length > 0
                  ? t("guardrails.decisionModel.model.placeholder")
                  : t("guardrails.decisionModel.model.placeholderEmpty")
              }
              className="w-full"
            />
            <ComboboxContent>
              <ComboboxEmpty>{t("guardrails.decisionModel.model.empty")}</ComboboxEmpty>
              <ComboboxList>
                {(model: string) => {
                  const provider = providerByModel.get(model);
                  return (
                    <ComboboxItem key={model} value={model} title={model}>
                      <span className="flex items-center gap-2">
                        {provider && <ProviderLogo provider={provider} />}
                        {model}
                      </span>
                    </ComboboxItem>
                  );
                }}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        )}
      </GuardrailField>

      {decisionModels.length === 0 && <NoDecisionModels providerFilter={providerFilter} />}

      <Field>
        <FieldLabel>{t("guardrails.decisionModel.questions.label")}</FieldLabel>
        <p className="m-0 text-xs text-muted-foreground">{t("guardrails.decisionModel.questions.description")}</p>
        {checks.map((check, index) => {
          const position = index + 1;
          return (
            <div key={check.id} className="relative space-y-3 rounded-lg border border-border bg-muted p-4">
              <button
                type="button"
                onClick={() => removeCheck(check.id)}
                aria-label={t("guardrails.decisionModel.question.remove", { position })}
                className="absolute top-2 right-2 p-1 text-muted-foreground transition-colors hover:text-destructive"
              >
                <X className="size-4" />
              </button>
              <div>
                <span className="mb-1 block text-xs font-medium text-muted-foreground">
                  {t("guardrails.decisionModel.question.nameLabel")}
                </span>
                <Input
                  aria-label={t("guardrails.decisionModel.question.nameAriaLabel", { position })}
                  placeholder={t("guardrails.decisionModel.question.namePlaceholder")}
                  className="bg-background"
                  value={check.name}
                  onChange={(event) => updateCheck(check.id, { name: event.target.value })}
                />
                {duplicateNames.has(check.name.trim()) && (
                  <p className="m-0 mt-1 text-xs text-destructive">
                    {t("guardrails.decisionModel.question.duplicateName")}
                  </p>
                )}
              </div>
              <div>
                <span className="mb-1 block text-xs font-medium text-muted-foreground">
                  {t("guardrails.decisionModel.question.questionLabel")}
                </span>
                <Textarea
                  aria-label={t("guardrails.decisionModel.question.questionAriaLabel", { position })}
                  rows={2}
                  placeholder={t("guardrails.decisionModel.question.questionPlaceholder")}
                  className="w-full resize-none bg-background"
                  value={check.instructions}
                  onChange={(event) => updateCheck(check.id, { instructions: event.target.value })}
                />
              </div>
              <div className="flex items-end gap-6">
                <div className="w-32">
                  <span className="mb-1 block text-xs font-medium text-muted-foreground">
                    {t("guardrails.decisionModel.question.actionLabel")}
                  </span>
                  <Select
                    items={actionItems}
                    value={check.action}
                    onValueChange={(next: string | null) =>
                      next && updateCheck(check.id, { action: next as "block" | "log" })
                    }
                  >
                    <SelectTrigger
                      className="w-full bg-background"
                      aria-label={t("guardrails.decisionModel.question.actionAriaLabel", { position })}
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {actionItems.map((item) => (
                        <SelectItem key={item.value} value={item.value}>
                          {item.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="w-56">
                  <span className="mb-1 block text-xs font-medium text-muted-foreground">
                    {t("guardrails.decisionModel.question.thresholdLabel")}
                  </span>
                  <div className="flex h-9 items-center">
                    <ThresholdSlider
                      aria-label={t("guardrails.decisionModel.question.thresholdAriaLabel", { position })}
                      value={check.threshold}
                      onChange={(next) => updateCheck(check.id, { threshold: next })}
                    />
                  </div>
                </div>
              </div>
            </div>
          );
        })}
        <div>
          <Button variant="outline" size="sm" onClick={addCheck}>
            <Plus className="size-3" />
            {t("guardrails.decisionModel.questions.add")}
          </Button>
        </div>
      </Field>

      <DecisionTestSection accessToken={accessToken} model={selectedModel} checks={checks} />
    </div>
  );
};

export default DecisionModelFields;
