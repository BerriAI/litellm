"use client";

import { Plus, X } from "lucide-react";
import React, { useState } from "react";
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
import type { DecisionModelCheckDraft } from "./buildDecisionModelParams";
import DecisionTestSection from "./DecisionTestSection";
import { DEFAULT_DECISION_THRESHOLD } from "./decisionModelQuestion";

export interface DecisionModelFieldsProps {
  accessToken: string | null;
  decisionProviders: string[];
  selectedProvider: string | null;
  onProviderChange: (provider: string | null) => void;
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
  decisionModels,
  checks,
  onChecksChange,
  control,
}) => {
  const [customName, setCustomName] = useState("");
  const [customInstructions, setCustomInstructions] = useState("");
  const [customAction, setCustomAction] = useState<"block" | "log">("block");
  const [customThreshold, setCustomThreshold] = useState(DEFAULT_DECISION_THRESHOLD);
  const [customNameTaken, setCustomNameTaken] = useState(false);
  const selectedModel = asText(useWatch({ control, name: "decision_model" }));

  const selectedNames = new Set(checks.map((check) => check.name));

  const setCheck = (name: string, patch: Partial<DecisionModelCheckDraft>) => {
    onChecksChange(checks.map((check) => (check.name === name ? { ...check, ...patch } : check)));
  };

  const addCustomCheck = () => {
    const name = customName.trim();
    const instructions = customInstructions.trim();
    if (!name || !instructions) return;
    if (selectedNames.has(name)) {
      setCustomNameTaken(true);
      return;
    }
    setCustomNameTaken(false);
    onChecksChange([
      ...checks,
      { name, instructions, action: customAction, threshold: customThreshold, enabled: true },
    ]);
    setCustomName("");
    setCustomInstructions("");
    setCustomAction("block");
    setCustomThreshold(DEFAULT_DECISION_THRESHOLD);
  };

  const removeCheck = (name: string) => onChecksChange(checks.filter((check) => check.name !== name));

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
          <Combobox items={decisionModels} value={asText(value) || null} onValueChange={onChange}>
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

      <FieldLabel>Questions</FieldLabel>

      {checks.length > 0 && (
      <div className="overflow-hidden rounded-lg border border-border">
        <div className="flex border-b border-border bg-muted/40 px-5 py-3">
          <span className="flex-1 font-semibold">Question</span>
          <span className="w-28 text-right font-semibold">Action</span>
          <span className="w-36 pl-4 font-semibold">Threshold</span>
          <span className="w-10" />
        </div>
        <div>
          {checks.map((check) => {
            const enabled = check.enabled !== false;
            return (
              <div
                key={check.name}
                className={`border-b border-border px-5 py-3 hover:bg-muted/40 ${enabled ? "bg-accent" : ""}`}
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="flex flex-1 items-start">
                    <Checkbox
                      className="mr-3 mt-0.5"
                      checked={enabled}
                      onCheckedChange={(next) => setCheck(check.name, { enabled: next === true })}
                      aria-label={check.name}
                    />
                    <div>
                      <span className={enabled ? "font-medium text-foreground" : "text-muted-foreground"}>
                        {check.name}
                      </span>
                      <p className="m-0 mt-0.5 text-xs text-muted-foreground">{check.instructions}</p>
                    </div>
                  </div>
                  <div className="w-28 shrink-0">
                    <Select
                      items={ACTION_ITEMS}
                      value={check.action}
                      onValueChange={(next: string | null) =>
                        next && setCheck(check.name, { action: next as "block" | "log" })
                      }
                      disabled={!enabled}
                    >
                      <SelectTrigger
                        className={`w-full ${enabled ? "" : "opacity-50"}`}
                        aria-label={`${check.name} action`}
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
                  <div className="w-36 shrink-0 pl-4">
                    <ThresholdSlider
                      aria-label={`${check.name} threshold`}
                      value={check.threshold}
                      onChange={(next) => setCheck(check.name, { threshold: next })}
                      disabled={!enabled}
                    />
                  </div>
                  <div className="flex w-10 shrink-0 justify-end">
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label={`Remove ${check.name}`}
                      className="size-6 text-destructive hover:text-destructive/80"
                      onClick={() => removeCheck(check.name)}
                    >
                      <X className="size-4" />
                    </Button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      )}

      <Field>
        <FieldLabel>Add a question</FieldLabel>
        <div className="space-y-2 rounded-md border border-dashed border-border p-3">
          <Input
            aria-label="Question name"
            placeholder="Question name (e.g. invoice_policy)"
            value={customName}
            onChange={(event) => {
              setCustomName(event.target.value);
              setCustomNameTaken(false);
            }}
          />
          <Textarea
            aria-label="Question"
            rows={2}
            placeholder="The question the decision model answers, e.g. Does the text ask about invoices?"
            className="w-full resize-none"
            value={customInstructions}
            onChange={(event) => setCustomInstructions(event.target.value)}
          />
          <div className="flex items-center gap-2">
            <div className="w-28">
              <Select
                items={ACTION_ITEMS}
                value={customAction}
                onValueChange={(next: string | null) => next && setCustomAction(next as "block" | "log")}
              >
                <SelectTrigger className="w-full" aria-label="New question action">
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
            <div className="w-36 pl-4">
              <ThresholdSlider
                aria-label="New question threshold"
                value={customThreshold}
                onChange={setCustomThreshold}
              />
            </div>
            <div className="flex-1" />
            <Button
              variant="outline"
              onClick={addCustomCheck}
              disabled={!customName.trim() || !customInstructions.trim()}
            >
              <Plus className="size-4" />
              Add question
            </Button>
          </div>
          {customNameTaken && (
            <p className="m-0 text-xs text-destructive">That name is already used by another question</p>
          )}
        </div>
      </Field>

      <DecisionTestSection accessToken={accessToken} model={selectedModel} checks={checks} />
    </div>
  );
};

export default DecisionModelFields;
