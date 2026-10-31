import React, { useId, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  SWITCHYARD_ALGORITHMS,
  SWITCHYARD_DEFAULTS,
  restoreSwitchyardDefaults,
  switchyardRecord,
  switchyardRequiredGroups,
  type SwitchyardAlgorithm,
  type SwitchyardDraft,
} from "./switchyard_config";

type Setting = {
  key: string;
  label: string;
  kind: "number" | "boolean" | "text" | "json" | "format" | "picker";
  hint?: string;
};
const CLASSIFIER_SETTINGS: Setting[] = [
  { key: "session_affinity", label: "Session affinity", kind: "boolean" },
  { key: "message_hash_fallback", label: "Message hash fallback", kind: "boolean" },
  { key: "recent_turn_window", label: "Recent turn window", kind: "number", hint: "Leave empty to use all turns" },
  { key: "max_output_tokens", label: "Classifier output token limit", kind: "number" },
];
const PROMPT_SETTINGS: Setting[] = [
  { key: "prompt", label: "Classifier prompt", kind: "text", hint: "Leave empty to use the built-in prompt" },
  { key: "response_format_type", label: "Classifier response format", kind: "format" },
];
const SETTINGS: Record<SwitchyardAlgorithm, Setting[]> = {
  random: [
    {
      key: "weights",
      label: "Model weights (JSON)",
      kind: "json",
      hint: "One weight per model in the any group in the order above. Leave empty for equal weights",
    },
    { key: "seed", label: "Random seed", kind: "number", hint: "Optional nonnegative integer" },
  ],
  capability: [
    {
      key: "base_threshold",
      label: "Capability threshold",
      kind: "number",
      hint: "LiteLLM preset 0.5, between 0 and 1",
    },
    { key: "threshold_step", label: "Threshold step", kind: "number" },
    ...CLASSIFIER_SETTINGS,
    ...PROMPT_SETTINGS,
  ],
  custom: [
    { key: "prompt", label: "Custom classifier prompt", kind: "text" },
    { key: "response_schema", label: "Response schema (JSON)", kind: "json" },
    { key: "selector", label: "Target selector", kind: "text" },
    { key: "default_target", label: "Default target group", kind: "text" },
    ...CLASSIFIER_SETTINGS,
  ],
  escalation: [
    { key: "confirmations", label: "Escalation confirmations", kind: "number" },
    { key: "recent_turn_window", label: "Recent turn window", kind: "number" },
    { key: "window_message_chars", label: "Characters per history message", kind: "number" },
    { key: "max_output_tokens", label: "Classifier output token limit", kind: "number" },
    ...PROMPT_SETTINGS,
    {
      key: "deescalation",
      label: "De-escalation (JSON)",
      kind: "json",
      hint: "Optional object with strong_min_calls, confirmations, strong_max_calls and weak_cooldown_calls. Leave empty to disable",
    },
  ],
  stage: [
    { key: "picker", label: "Stage picker", kind: "picker" },
    {
      key: "confidence_threshold",
      label: "Confidence threshold",
      kind: "number",
      hint: "LiteLLM preset 0.5, between 0 and 1",
    },
    { key: "recent_window", label: "Recent stage window", kind: "number" },
    { key: "escalation_note", label: "Escalation note", kind: "text" },
    { key: "deescalation_note", label: "De-escalation note", kind: "text" },
    { key: "only_on_wrong_signal_escalation", label: "Escalate only on wrong signal", kind: "boolean" },
    { key: "capable_system_prompt", label: "Capable model system prompt", kind: "text" },
    { key: "efficient_system_prompt", label: "Efficient model system prompt", kind: "text" },
    {
      key: "tool_semantics",
      label: "Tool semantics (JSON)",
      kind: "json",
      hint: "Map tool categories to lists of tool names",
    },
    {
      key: "classifier",
      label: "Fallback capability classifier (JSON)",
      kind: "json",
      hint: "Optional capability configuration. Requires a judge model group",
    },
  ],
};

function SettingInput({
  setting,
  id,
  value,
  onChange,
}: {
  setting: Setting;
  id: string;
  value: unknown;
  onChange: (next: unknown) => void;
}) {
  if (setting.kind === "boolean")
    return (
      <input id={id} type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />
    );
  if (setting.kind === "number")
    return (
      <Input
        id={id}
        type="number"
        step="any"
        value={typeof value === "number" ? value : ""}
        onChange={(event) => onChange(event.target.value === "" ? null : Number(event.target.value))}
      />
    );
  if (setting.kind === "format" || setting.kind === "picker") {
    const options = setting.kind === "format" ? ["json_schema", "json_object"] : ["efficient_first", "capable_first"];
    return (
      <Select value={String(value ?? "")} onValueChange={(next) => next && onChange(next)}>
        <SelectTrigger id={id}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option} value={option}>
              {option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  }
  const textValue = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return (
    <Textarea
      id={id}
      value={value == null ? "" : textValue}
      onChange={(event) => onChange(event.target.value || null)}
    />
  );
}

export default function SwitchyardConfig({
  value,
  onChange,
  modelNames = [],
}: {
  value: SwitchyardDraft;
  onChange: (value: SwitchyardDraft) => void;
  modelNames?: string[];
}) {
  const id = useId();
  const [newGroup, setNewGroup] = useState("");
  const algorithm = value.algorithm as SwitchyardAlgorithm;
  const knownAlgorithm = SWITCHYARD_ALGORITHMS.includes(algorithm);
  const config = algorithm === "random" ? value : switchyardRecord(value[algorithm]);
  const defaults = SWITCHYARD_DEFAULTS[algorithm] ?? {};
  const models = switchyardRecord(value.models);
  const requiredGroups = switchyardRequiredGroups(algorithm, Boolean(switchyardRecord(value.stage).classifier));
  const groups = Array.from(new Set([...requiredGroups, ...Object.keys(models)]));
  const update = (key: string, next: unknown) =>
    onChange(
      algorithm === "random" ? { ...value, [key]: next } : { ...value, [algorithm]: { ...config, [key]: next } },
    );
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor={`${id}-algorithm`}>Switchyard algorithm</Label>
        <Select
          value={String(value.algorithm ?? "")}
          onValueChange={(next) => next && onChange({ ...value, algorithm: next })}
        >
          <SelectTrigger id={`${id}-algorithm`} className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SWITCHYARD_ALGORITHMS.map((item) => (
              <SelectItem key={item} value={item}>
                {item.charAt(0).toUpperCase() + item.slice(1)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Switching algorithms keeps your other settings. Model names must refer to configured LiteLLM models.
        </p>
      </div>
      <p className="text-xs text-muted-foreground">
        Supports chat completions. Escalation may buffer responses before streaming the accepted answer.
      </p>
      <fieldset className="space-y-3">
        <legend className="font-medium">Model groups</legend>
        <p className="text-xs text-muted-foreground">
          Enter ordered, comma-separated model names for each group. Later models are fallback candidates. Stage routing
          only needs a judge when its fallback classifier is enabled.
        </p>
        <datalist id={`${id}-models`}>
          {modelNames.map((name) => (
            <option key={name} value={name} />
          ))}
        </datalist>
        {groups.map((group) => (
          <div key={group} className="space-y-1">
            <Label htmlFor={`${id}-group-${group}`}>{group} models</Label>
            <div className="flex gap-2">
              <Input
                id={`${id}-group-${group}`}
                list={`${id}-models`}
                value={Array.isArray(models[group]) ? models[group].join(", ") : ""}
                placeholder="model-name, fallback-model"
                onChange={(event) =>
                  onChange({
                    ...value,
                    models: { ...models, [group]: event.target.value.split(",").map((name) => name.trim()) },
                  })
                }
              />
              {!requiredGroups.includes(group) && (
                <Button
                  type="button"
                  variant="outline"
                  aria-label={`Remove ${group} group`}
                  onClick={() =>
                    onChange({
                      ...value,
                      models: Object.fromEntries(Object.entries(models).filter(([name]) => name !== group)),
                    })
                  }
                >
                  Remove
                </Button>
              )}
            </div>
          </div>
        ))}
        <div className="flex gap-2">
          <Input
            aria-label="New model group name"
            value={newGroup}
            onChange={(event) => setNewGroup(event.target.value)}
            placeholder="New target group name"
          />
          <Button
            type="button"
            variant="outline"
            disabled={!newGroup.trim() || groups.includes(newGroup.trim())}
            onClick={() => {
              onChange({ ...value, models: { ...models, [newGroup.trim()]: [] } });
              setNewGroup("");
            }}
          >
            Add group
          </Button>
        </div>
      </fieldset>
      <details className="rounded-md border p-4">
        <summary className="cursor-pointer font-medium">Advanced settings</summary>
        <div className="mt-4 space-y-4">
          <p className="text-xs text-muted-foreground">
            Switchyard defaults are shown unless noted as LiteLLM presets. Empty optional fields use the default
            behavior.
          </p>
          <Label>
            <input
              type="checkbox"
              checked={value.fallback_enabled !== false}
              onChange={(event) => onChange({ ...value, fallback_enabled: event.target.checked })}
            />{" "}
            Enable model failure fallback
          </Label>
          {(SETTINGS[algorithm] ?? []).map((setting) => {
            const current = config[setting.key] === undefined ? defaults[setting.key] : config[setting.key];
            const fieldId = `${id}-${setting.key}`;
            return (
              <div key={setting.key} className="space-y-1">
                <Label htmlFor={fieldId}>{setting.label}</Label>
                <SettingInput
                  setting={setting}
                  id={fieldId}
                  value={current}
                  onChange={(next) => update(setting.key, next)}
                />
                {setting.hint && <p className="text-xs text-muted-foreground">{setting.hint}</p>}
                {setting.key === "prompt" && algorithm !== "custom" && (
                  <Button type="button" variant="link" onClick={() => update("prompt", null)}>
                    Restore built-in prompt
                  </Button>
                )}
              </div>
            );
          })}
          <Button
            type="button"
            variant="outline"
            disabled={!knownAlgorithm}
            onClick={() => onChange(restoreSwitchyardDefaults(value))}
          >
            Restore defaults
          </Button>
        </div>
      </details>
    </div>
  );
}
