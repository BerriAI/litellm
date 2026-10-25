"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { Info, Plus, X } from "lucide-react";
import React from "react";
import { useWatch } from "react-hook-form";
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

const ACTION_ITEMS = [
  { label: "Block", value: "block" },
  { label: "Log only", value: "log" },
];

const ProviderLogo: React.FC<{ provider: string }> = ({ provider }) => (
  <Logo provider={provider} label={getProviderLogoAndName(provider).displayName} className="size-4 shrink-0" />
);

const NoDecisionModels: React.FC<{ providerFilter: string | null }> = ({ providerFilter }) => {
  const { t } = useLanguage();
  const source = providerFilter ? ` from the ${getProviderLogoAndName(providerFilter).displayName} provider` : "";
  return (
    <Alert variant="info" role="note" aria-label={t("No decision models")}>
      <Info />
      <AlertTitle>{t("No decision models yet")}</AlertTitle>
      <AlertDescription>
        <p>
          {t("Add a decision model")}
          {source} {t("to use this guardrail. A model shows up here once its mode is decisions.")}{" "}
          <a href={uiHref("models-and-endpoints")} target="_blank" rel="noopener noreferrer">
            {t("Add a model")}
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
  const { t } = useLanguage();
  const selectedModel = asText(useWatch({ control, name: "decision_model" }));
  const duplicateNames = duplicateDecisionCheckNames(checks);
  const providerByModel = new Map(decisionModels.map((option) => [option.model, option.provider]));

  const updateCheck = (id: string, patch: Partial<DecisionModelCheckDraft>) =>
    onChecksChange(checks.map((check) => (check.id === id ? { ...check, ...patch } : check)));

  const addCheck = () => onChecksChange([...checks, newDecisionCheckDraft(checks)]);

  const removeCheck = (id: string) => onChecksChange(checks.filter((check) => check.id !== id));

  return (
    <div className="space-y-5">
      <div className="rounded-md border border-success/20 bg-success/10 px-3.5 py-2.5 text-[13px] text-success">
        {t("The")} <strong>{t("Decision Model")}</strong>{" "}
        {t(
          "answers yes/no questions about each request (pre_call, during_call) or response (post_call). A question whose probability reaches its threshold blocks the call, or is recorded when set to Log only.",
        )}
      </div>

      <GuardrailField
        control={control}
        name="decision_model"
        label={labelWithHint(
          t("Decision Model"),
          t("A decisions-API model on this proxy, such as typesafe/jev-latest. It scores every question below."),
        )}
        rules={requiredRule(t("Select a decision model"))}
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
              placeholder={decisionModels.length > 0 ? t("Select a decision model") : t("No decision models yet")}
              className="w-full"
            />
            <ComboboxContent>
              <ComboboxEmpty>{t("No matching models")}</ComboboxEmpty>
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
        <FieldLabel>{t("Questions")}</FieldLabel>
        <p className="m-0 text-xs text-muted-foreground">
          {t(
            "Yes/no questions the decision model answers about the text. Block stops the call once a score reaches the threshold, and Log only records it.",
          )}
        </p>
        {checks.map((check, index) => {
          const position = index + 1;
          return (
            <div key={check.id} className="relative space-y-3 rounded-lg border border-border bg-muted p-4">
              <button
                type="button"
                onClick={() => removeCheck(check.id)}
                aria-label={t("Remove question {position}", { position: String(position) })}
                className="absolute top-2 right-2 p-1 text-muted-foreground transition-colors hover:text-destructive"
              >
                <X className="size-4" />
              </button>
              <div>
                <span className="mb-1 block text-xs font-medium text-muted-foreground">{t("Name")}</span>
                <Input
                  aria-label={t("Question {position} name", { position: String(position) })}
                  placeholder={t("Name (e.g. invoice_policy)")}
                  className="bg-background"
                  value={check.name}
                  onChange={(event) => updateCheck(check.id, { name: event.target.value })}
                />
                {duplicateNames.has(check.name.trim()) && (
                  <p className="m-0 mt-1 text-xs text-destructive">{t("Another question already uses this name")}</p>
                )}
              </div>
              <div>
                <span className="mb-1 block text-xs font-medium text-muted-foreground">{t("Question")}</span>
                <Textarea
                  aria-label={t("Question {position}", { position: String(position) })}
                  rows={2}
                  placeholder={t("The question the decision model answers, e.g. Does the text ask about invoices?")}
                  className="w-full resize-none bg-background"
                  value={check.instructions}
                  onChange={(event) => updateCheck(check.id, { instructions: event.target.value })}
                />
              </div>
              <div className="flex items-end gap-6">
                <div className="w-32">
                  <span className="mb-1 block text-xs font-medium text-muted-foreground">{t("Action")}</span>
                  <Select
                    items={ACTION_ITEMS}
                    value={check.action}
                    onValueChange={(next: string | null) =>
                      next && updateCheck(check.id, { action: next as "block" | "log" })
                    }
                  >
                    <SelectTrigger
                      className="w-full bg-background"
                      aria-label={t("Question {position} action", { position: String(position) })}
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {ACTION_ITEMS.map((item) => (
                        <SelectItem key={item.value} value={item.value}>
                          {item.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="w-56">
                  <span className="mb-1 block text-xs font-medium text-muted-foreground">{t("Threshold")}</span>
                  <div className="flex h-9 items-center">
                    <ThresholdSlider
                      aria-label={t("Question {position} threshold", { position: String(position) })}
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
            {t("Add question")}
          </Button>
        </div>
      </Field>

      <DecisionTestSection accessToken={accessToken} model={selectedModel} checks={checks} />
    </div>
  );
};

export default DecisionModelFields;
