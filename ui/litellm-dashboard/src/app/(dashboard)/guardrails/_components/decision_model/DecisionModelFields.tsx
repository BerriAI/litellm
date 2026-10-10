"use client";

import { Plus, X } from "lucide-react";
import React from "react";
import { useWatch } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
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
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { asText, GuardrailField, labelWithHint, requiredRule, type GuardrailFormControl } from "../GuardrailFormField";
import { duplicateDecisionCheckNames, type DecisionModelCheckDraft } from "./buildDecisionModelParams";
import DecisionTestSection from "./DecisionTestSection";
import { newDecisionCheckDraft } from "./decisionModelQuestion";

export interface DecisionModelFieldsProps {
  accessToken: string | null;
  decisionProviders: string[];
  selectedProvider: string | null;
  onProviderChange: (provider: string | null) => void;
  onModelChange: () => void;
  decisionModels: string[];
  checks: DecisionModelCheckDraft[];
  onChecksChange: (checks: DecisionModelCheckDraft[]) => void;
  control: GuardrailFormControl;
}

const ACTION_ITEMS = [
  { label: "Block", value: "block" },
  { label: "Log only", value: "log" },
];

const ThresholdSlider: React.FC<{
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  "aria-label": string;
}> = ({ value, onChange, disabled, "aria-label": ariaLabel }) => (
  <div className="flex items-center gap-2">
    <Slider
      aria-label={ariaLabel}
      min={0}
      max={1}
      step={0.01}
      value={[value]}
      onValueChange={(next) => onChange(Array.isArray(next) ? next[0] ?? 0 : next)}
      disabled={disabled}
      className="min-w-[120px] flex-1"
    />
    <span className="w-9 whitespace-nowrap text-right text-xs tabular-nums text-muted-foreground">
      {value.toFixed(2)}
    </span>
  </div>
);

const DecisionModelFields: React.FC<DecisionModelFieldsProps> = ({
  accessToken,
  decisionProviders,
  selectedProvider,
  onProviderChange,
  onModelChange,
  decisionModels,
  checks,
  onChecksChange,
  control,
}) => {
  const selectedModel = asText(useWatch({ control, name: "decision_model" }));
  const duplicateNames = duplicateDecisionCheckNames(checks);

  const updateCheck = (id: string, patch: Partial<DecisionModelCheckDraft>) =>
    onChecksChange(checks.map((check) => (check.id === id ? { ...check, ...patch } : check)));

  const addCheck = () => onChecksChange([...checks, newDecisionCheckDraft(checks)]);

  const removeCheck = (id: string) => onChecksChange(checks.filter((check) => check.id !== id));

  return (
    <div className="space-y-5">
      <div className="rounded-md border border-success/20 bg-success/10 px-3.5 py-2.5 text-[13px] text-success">
        The <strong>Decision Model</strong> answers yes/no questions about each request (pre_call, during_call) or
        response (post_call). A question whose probability reaches its threshold blocks the call, or is recorded when
        set to Log only.
      </div>

      <Field>
        <FieldLabel>Decision Provider</FieldLabel>
        <Select
          items={decisionProviders.map((provider) => {
            const { displayName } = getProviderLogoAndName(provider);
            return { label: displayName, value: provider };
          })}
          value={selectedProvider}
          onValueChange={(next: string | null) => onProviderChange(next)}
        >
          <SelectTrigger className="w-full" aria-label="Decision Provider">
            <SelectValue placeholder="Select a provider" />
          </SelectTrigger>
          <SelectContent>
            {decisionProviders.map((provider) => {
              const { logo, displayName } = getProviderLogoAndName(provider);
              return (
                <SelectItem key={provider} value={provider}>
                  <span className="flex items-center gap-2">
                    {logo && <img src={logo} alt={`${displayName} logo`} className="size-4" />}
                    {displayName}
                  </span>
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
      </Field>

      <GuardrailField
        control={control}
        name="decision_model"
        label={labelWithHint(
          "Decision Model",
          "A decisions-API model on this proxy, such as typesafe/jev-latest. It scores every question below.",
        )}
        rules={requiredRule("Select a decision model")}
      >
        {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
          <Combobox
            items={decisionModels}
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
              placeholder="Select a decision model"
              className="w-full"
            />
            <ComboboxContent>
              <ComboboxEmpty>No matching models</ComboboxEmpty>
              <ComboboxList>
                {(model: string) => (
                  <ComboboxItem key={model} value={model} title={model}>
                    {model}
                  </ComboboxItem>
                )}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        )}
      </GuardrailField>

      {selectedProvider && decisionModels.length === 0 && (
        <p className="text-sm text-muted-foreground">
          No decision models for this provider. Set model_info.mode: evaluation on a deployment to list it here.
        </p>
      )}

      <Field>
        <FieldLabel>Questions</FieldLabel>
        <p className="m-0 text-xs text-muted-foreground">
          Yes/no questions the decision model answers about the text. Block stops the call once a score reaches the
          threshold, and Log only records it.
        </p>
        {checks.map((check, index) => {
          const position = index + 1;
          const enabled = check.enabled !== false;
          return (
            <div
              key={check.id}
              className={`relative space-y-2 rounded-lg border border-border bg-muted p-4 ${enabled ? "" : "opacity-60"}`}
            >
              <button
                type="button"
                onClick={() => removeCheck(check.id)}
                aria-label={`Remove question ${position}`}
                className="absolute top-2 right-2 p-1 text-muted-foreground transition-colors hover:text-destructive"
              >
                <X className="size-4" />
              </button>
              <div className="flex items-center gap-3 pr-8">
                <Checkbox
                  checked={enabled}
                  onCheckedChange={(next) => updateCheck(check.id, { enabled: next === true })}
                  aria-label={`Enable question ${position}`}
                />
                <Input
                  aria-label={`Question ${position} name`}
                  placeholder="Name (e.g. invoice_policy)"
                  className="bg-background"
                  value={check.name}
                  onChange={(event) => updateCheck(check.id, { name: event.target.value })}
                />
              </div>
              {enabled && duplicateNames.has(check.name.trim()) && (
                <p className="m-0 text-xs text-destructive">Another question already uses this name</p>
              )}
              <Textarea
                aria-label={`Question ${position}`}
                rows={2}
                placeholder="The question the decision model answers, e.g. Does the text ask about invoices?"
                className="w-full resize-none bg-background"
                value={check.instructions}
                onChange={(event) => updateCheck(check.id, { instructions: event.target.value })}
              />
              <div className="flex items-end gap-6">
                <div className="w-32">
                  <span className="mb-1 block text-xs font-medium text-muted-foreground">Action</span>
                  <Select
                    items={ACTION_ITEMS}
                    value={check.action}
                    onValueChange={(next: string | null) =>
                      next && updateCheck(check.id, { action: next as "block" | "log" })
                    }
                    disabled={!enabled}
                  >
                    <SelectTrigger className="w-full bg-background" aria-label={`Question ${position} action`}>
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
                  <span className="mb-1 block text-xs font-medium text-muted-foreground">Threshold</span>
                  <div className="flex h-9 items-center">
                    <ThresholdSlider
                      aria-label={`Question ${position} threshold`}
                      value={check.threshold}
                      onChange={(next) => updateCheck(check.id, { threshold: next })}
                      disabled={!enabled}
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
            Add question
          </Button>
        </div>
      </Field>

      <DecisionTestSection accessToken={accessToken} model={selectedModel} checks={checks} />
    </div>
  );
};

export default DecisionModelFields;
