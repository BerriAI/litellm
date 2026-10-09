"use client";

import { Plus, X } from "lucide-react";
import React, { useId, useState } from "react";
import { Badge } from "@/components/ui/badge";
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
import { Textarea } from "@/components/ui/textarea";
import { asText, GuardrailField, labelWithHint, requiredRule, type GuardrailFormControl } from "../GuardrailFormField";
import type { DecisionModelCheckDraft } from "./buildDecisionModelParams";

export interface DecisionModelCheckPreset {
  name: string;
  label: string;
  instructions: string;
}

export interface DecisionModelFieldsProps {
  decisionModels: string[];
  presets: DecisionModelCheckPreset[];
  checks: DecisionModelCheckDraft[];
  onChecksChange: (checks: DecisionModelCheckDraft[]) => void;
  control: GuardrailFormControl;
}

const ACTION_ITEMS = [
  { label: "Block", value: "block" },
  { label: "Log only", value: "log" },
];

const DecisionModelFields: React.FC<DecisionModelFieldsProps> = ({
  decisionModels,
  presets,
  checks,
  onChecksChange,
  control,
}) => {
  const nameId = useId();
  const [customName, setCustomName] = useState("");
  const [customInstructions, setCustomInstructions] = useState("");
  const [customNameTaken, setCustomNameTaken] = useState(false);

  const selectedNames = new Set(checks.map((check) => check.name));

  const setCheck = (name: string, patch: Partial<DecisionModelCheckDraft>) => {
    onChecksChange(checks.map((check) => (check.name === name ? { ...check, ...patch } : check)));
  };

  const togglePreset = (preset: DecisionModelCheckPreset, selected: boolean) => {
    if (selected) {
      onChecksChange([...checks, { name: preset.name, label: preset.label, action: "block", threshold: 0.5 }]);
    } else {
      onChecksChange(checks.filter((check) => check.name !== preset.name));
    }
  };

  const selectAll = () => {
    const missing = presets.filter((preset) => !selectedNames.has(preset.name));
    onChecksChange([
      ...checks,
      ...missing.map((preset) => ({
        name: preset.name,
        label: preset.label,
        action: "block" as const,
        threshold: 0.5,
      })),
    ]);
  };

  const addCustomCheck = () => {
    const name = customName.trim();
    const instructions = customInstructions.trim();
    if (!name || !instructions) return;
    if (selectedNames.has(name) || presets.some((preset) => preset.name === name)) {
      setCustomNameTaken(true);
      return;
    }
    setCustomNameTaken(false);
    onChecksChange([...checks, { name, instructions, action: "block", threshold: 0.5, custom: true }]);
    setCustomName("");
    setCustomInstructions("");
  };

  const removeCheck = (name: string) => onChecksChange(checks.filter((check) => check.name !== name));

  return (
    <div className="space-y-5">
      <div className="rounded-md border border-success/20 bg-success/10 px-3.5 py-2.5 text-[13px] text-success">
        The <strong>Decision Model</strong> answers yes/no questions about each request (pre_call, during_call) or
        response (post_call). A check whose probability reaches its threshold blocks the call, or is recorded when set
        to Log only.
      </div>

      <GuardrailField
        control={control}
        name="decision_model"
        label={labelWithHint(
          "Decision Model",
          "A model on this proxy whose mode is 'evaluation', such as typesafe/jev-latest. It scores every check below.",
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

      {decisionModels.length === 0 && (
        <p className="text-sm text-muted-foreground">
          No decision models yet. Add one with mode <code>evaluation</code>, such as <code>typesafe/jev-latest</code>,
          on the Models page.
        </p>
      )}

      <div className="flex items-center justify-between">
        <FieldLabel>Checks</FieldLabel>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={selectAll}>
            Select all
          </Button>
          <Button variant="outline" size="sm" onClick={() => onChecksChange(checks.filter((c) => c.custom))}>
            Unselect all
          </Button>
        </div>
      </div>

      <div className="overflow-hidden rounded-lg border border-border">
        <div className="flex border-b border-border bg-muted/40 px-5 py-3">
          <span className="flex-1 font-semibold">Check</span>
          <span className="w-28 text-right font-semibold">Action</span>
          <span className="w-28 pl-4 font-semibold">Threshold</span>
        </div>
        <div>
          {presets.map((preset) => {
            const check = checks.find((entry) => entry.name === preset.name);
            const selected = check !== undefined;
            return (
              <div
                key={preset.name}
                className={`flex items-center justify-between border-b border-border px-5 py-3 hover:bg-muted/40 ${
                  selected ? "bg-accent" : ""
                }`}
              >
                <div className="flex flex-1 items-start">
                  <Checkbox
                    className="mr-3 mt-0.5"
                    checked={selected}
                    onCheckedChange={(next) => togglePreset(preset, next === true)}
                    aria-label={preset.label}
                  />
                  <div>
                    <span className={selected ? "font-medium text-foreground" : "text-muted-foreground"}>
                      {preset.label}
                    </span>
                    <p className="m-0 mt-0.5 text-xs text-muted-foreground">{preset.instructions}</p>
                  </div>
                </div>
                <div className="w-28">
                  <Select
                    items={ACTION_ITEMS}
                    value={selected ? check.action : "block"}
                    onValueChange={(next: string | null) =>
                      next && setCheck(preset.name, { action: next as "block" | "log" })
                    }
                    disabled={!selected}
                  >
                    <SelectTrigger
                      className={`w-full ${selected ? "" : "opacity-50"}`}
                      aria-label={`${preset.label} action`}
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
                <div className="w-28 pl-4">
                  <Input
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    disabled={!selected}
                    aria-label={`${preset.label} threshold`}
                    value={selected ? check.threshold : 0.5}
                    onChange={(event) => setCheck(preset.name, { threshold: Number(event.target.value) || 0 })}
                  />
                </div>
              </div>
            );
          })}
          {checks
            .filter((check) => check.custom)
            .map((check) => (
              <div
                key={check.name}
                className="flex items-center justify-between border-b border-border bg-accent px-5 py-3 hover:bg-muted/40"
              >
                <div className="flex flex-1 items-start">
                  <div>
                    <span className="font-medium text-foreground">{check.name}</span>
                    <Badge variant="secondary" className="ml-2">
                      custom
                    </Badge>
                    <p className="m-0 mt-0.5 text-xs text-muted-foreground">{check.instructions}</p>
                  </div>
                </div>
                <div className="w-28">
                  <Select
                    items={ACTION_ITEMS}
                    value={check.action}
                    onValueChange={(next: string | null) =>
                      next && setCheck(check.name, { action: next as "block" | "log" })
                    }
                  >
                    <SelectTrigger className="w-full" aria-label={`${check.name} action`}>
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
                <div className="w-28 pl-4">
                  <Input
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    aria-label={`${check.name} threshold`}
                    value={check.threshold}
                    onChange={(event) => setCheck(check.name, { threshold: Number(event.target.value) || 0 })}
                  />
                </div>
                <div>
                  <Button
                    variant="ghost"
                    size="sm"
                    aria-label={`Remove ${check.name}`}
                    className="text-destructive hover:text-destructive/80"
                    onClick={() => removeCheck(check.name)}
                  >
                    <X className="size-4" />
                  </Button>
                </div>
              </div>
            ))}
        </div>
      </div>

      <Field>
        <FieldLabel>Add a custom check</FieldLabel>
        <div className="rounded-md border border-dashed border-border p-3">
          <div className="flex gap-2">
            <Input
              id={nameId}
              aria-label="Custom check name"
              placeholder="Check name (e.g. invoice_policy)"
              value={customName}
              onChange={(event) => {
                setCustomName(event.target.value);
                setCustomNameTaken(false);
              }}
            />
            <Button
              variant="outline"
              onClick={addCustomCheck}
              disabled={!customName.trim() || !customInstructions.trim()}
            >
              <Plus className="size-4" />
              Add check
            </Button>
          </div>
          {customNameTaken && (
            <p className="mt-1 text-xs text-destructive">That name is already used by another check</p>
          )}
          <Textarea
            aria-label="Custom check instructions"
            rows={2}
            placeholder="The question the decision model answers, e.g. Does the text ask about invoices?"
            className="mt-2 w-full resize-none"
            value={customInstructions}
            onChange={(event) => setCustomInstructions(event.target.value)}
          />
        </div>
      </Field>
    </div>
  );
};

export default DecisionModelFields;
