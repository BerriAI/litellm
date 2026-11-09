import { SearchSelect } from "@/components/shared/SearchSelect";
import { Field, FieldLabel } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import { InputGroup, InputGroupAddon, InputGroupInput, InputGroupText } from "@/components/ui/input-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import i18n from "@/i18n/i18n";
import { Plus, X } from "lucide-react";
import React, { useState } from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";

export interface ModelBudgetConfig {
  budget_limit: number;
  time_period: string;
  /** BudgetConfig also carries tpm_limit and rpm_limit, which this editor does not model. */
  [passthrough: string]: unknown;
}

export type ModelMaxBudget = Record<string, ModelBudgetConfig>;

export interface ModelBudgetUsage {
  current_spend: number;
  budget_limit: number | null;
  time_period: string | null;
}

interface ModelBudgetEntry {
  id: string;
  model: string | null;
  budgetLimit: number | null;
  timePeriod: string;
  extra: Readonly<Record<string, unknown>>;
}

// BudgetConfig aliases budget_limit onto max_budget and time_period onto
// budget_duration, and the proxy stores whichever spelling the client sent.
const MODELLED_FIELDS: readonly string[] = ["budget_limit", "time_period", "max_budget", "budget_duration"];

const readNumber = (raw: unknown): number | null => {
  const value = typeof raw === "string" ? Number(raw) : raw;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
};

const readPeriod = (raw: unknown): string | null => (typeof raw === "string" && raw !== "" ? raw : null);

export const MODEL_BUDGET_PERIOD_OPTIONS: readonly { value: string; labelKey: ParseKeys }[] = [
  { value: "1h", labelKey: "keyTeamHelpers.modelMaxBudgetEditor.periods.hourly" },
  { value: "24h", labelKey: "keyTeamHelpers.modelMaxBudgetEditor.periods.daily" },
  { value: "7d", labelKey: "keyTeamHelpers.modelMaxBudgetEditor.periods.weekly" },
  { value: "30d", labelKey: "keyTeamHelpers.modelMaxBudgetEditor.periods.monthly" },
  { value: "1mo", labelKey: "keyTeamHelpers.modelMaxBudgetEditor.periods.calendarMonth" },
];

const DEFAULT_PERIOD = "30d";

const MODEL_PATTERN_VALUE = {
  hintKey: "keyTeamHelpers.modelMaxBudgetEditor.modelPattern" as ParseKeys,
  accepts: (typed: string) => typed.includes("*"),
};

export const entriesToModelMaxBudget = (entries: readonly ModelBudgetEntry[]): ModelMaxBudget =>
  Object.fromEntries(
    entries
      .filter(
        (entry): entry is ModelBudgetEntry & { model: string; budgetLimit: number } =>
          entry.model !== null && entry.budgetLimit !== null,
      )
      .map((entry) => [
        entry.model,
        { ...entry.extra, budget_limit: entry.budgetLimit, time_period: entry.timePeriod },
      ]),
  );

export const modelMaxBudgetToEntries = (budget: ModelMaxBudget | null | undefined): ModelBudgetEntry[] =>
  Object.entries(budget ?? {}).map(([model, config], index) => ({
    id: `existing-${index}`,
    model,
    budgetLimit: readNumber(config?.budget_limit) ?? readNumber(config?.max_budget),
    timePeriod: readPeriod(config?.time_period) ?? readPeriod(config?.budget_duration) ?? DEFAULT_PERIOD,
    extra: Object.fromEntries(Object.entries(config ?? {}).filter(([field]) => !MODELLED_FIELDS.includes(field))),
  }));

export const MODEL_MAX_BUDGET_PREMIUM_HINT: ParseKeys = "keyTeamHelpers.modelMaxBudgetEditor.premiumHint";

interface ModelMaxBudgetEditorProps {
  value: ModelMaxBudget;
  onChange: (value: ModelMaxBudget) => void;
  availableModels: string[];
  /** The proxy rejects a populated model_max_budget without an enterprise license. */
  premiumUser: boolean;
  usage?: Record<string, ModelBudgetUsage> | null;
}

export function ModelMaxBudgetEditor({
  value,
  onChange,
  availableModels,
  premiumUser,
  usage,
}: ModelMaxBudgetEditorProps) {
  const { t } = useTranslation();
  const [entries, setEntries] = useState<ModelBudgetEntry[]>(() => modelMaxBudgetToEntries(value));

  const incoming = JSON.stringify(value ?? {});
  const [syncedValue, setSyncedValue] = useState(incoming);
  if (incoming !== syncedValue) {
    setSyncedValue(incoming);
    if (incoming !== JSON.stringify(entriesToModelMaxBudget(entries))) setEntries(modelMaxBudgetToEntries(value));
  }

  const emitChange = (updated: ModelBudgetEntry[]) => {
    setEntries(updated);
    onChange(entriesToModelMaxBudget(updated));
  };

  const addEntry = () =>
    emitChange([
      ...entries,
      { id: Date.now().toString(), model: null, budgetLimit: null, timePeriod: DEFAULT_PERIOD, extra: {} },
    ]);

  const removeEntry = (id: string) => emitChange(entries.filter((entry) => entry.id !== id));

  const updateEntry = (id: string, patch: Partial<ModelBudgetEntry>) =>
    emitChange(entries.map((entry) => (entry.id === id ? { ...entry, ...patch } : entry)));

  const usedModels = new Set(entries.map((entry) => entry.model).filter(Boolean));
  const hintWhenLocked = premiumUser ? undefined : t(MODEL_MAX_BUDGET_PREMIUM_HINT);
  const periodItems = MODEL_BUDGET_PERIOD_OPTIONS.map(({ value, labelKey }) => ({ value, label: t(labelKey) }));
  const modelPatternValue = { hint: t(MODEL_PATTERN_VALUE.hintKey), accepts: MODEL_PATTERN_VALUE.accepts };

  const blurb = (
    <div className="text-xs text-muted-foreground">
      {premiumUser ? t("keyTeamHelpers.modelMaxBudgetEditor.blurb") : t(MODEL_MAX_BUDGET_PREMIUM_HINT)}
    </div>
  );

  if (entries.length === 0) {
    return (
      <div>
        <div className="mb-2">{blurb}</div>
        <Button variant="outline" size="sm" onClick={addEntry} disabled={!premiumUser} title={hintWhenLocked}>
          <Plus className="w-3 h-3" />
          {t("keyTeamHelpers.modelMaxBudgetEditor.addModelBudget")}
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {blurb}
      {entries.map((entry) => {
        const modelOptions = availableModels.filter((model) => model === entry.model || !usedModels.has(model));
        const spent = entry.model ? usage?.[entry.model]?.current_spend : undefined;
        return (
          <div key={entry.id} className="relative rounded-lg border border-border bg-muted p-4">
            <button
              type="button"
              onClick={() => removeEntry(entry.id)}
              disabled={!premiumUser}
              title={hintWhenLocked}
              aria-label={t("keyTeamHelpers.modelMaxBudgetEditor.removeModelBudget")}
              className="absolute top-2 right-2 text-muted-foreground hover:text-destructive transition-colors p-1"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="mb-3">
              <label className="block text-xs font-medium text-muted-foreground mb-1">
                {t("keyTeamHelpers.modelMaxBudgetEditor.model")}
              </label>
              <SearchSelect
                options={modelOptions.map((model) => ({ label: model, value: model }))}
                value={entry.model}
                onValueChange={(model) => updateEntry(entry.id, { model })}
                placeholder={t("keyTeamHelpers.modelMaxBudgetEditor.selectModel")}
                emptyText={t("keyTeamHelpers.modelMaxBudgetEditor.noModels")}
                customValue={modelPatternValue}
                reservedValues={entries.flatMap((other) => (other.id !== entry.id && other.model ? [other.model] : []))}
                disabled={!premiumUser}
              />
            </div>

            <div className="flex gap-2 items-center">
              <InputGroup className="w-40">
                <InputGroupAddon>
                  <InputGroupText>$</InputGroupText>
                </InputGroupAddon>
                <InputGroupInput
                  type="number"
                  // A per-model cap is often a fraction of a cent, so a 0.01
                  // step would make the browser refuse the value on submit.
                  step="any"
                  min={0}
                  value={entry.budgetLimit ?? ""}
                  onChange={(event) => {
                    const typed = event.target.valueAsNumber;
                    updateEntry(entry.id, { budgetLimit: Number.isNaN(typed) ? null : typed });
                  }}
                  placeholder={t("keyTeamHelpers.modelMaxBudgetEditor.maxSpend")}
                  disabled={!premiumUser}
                />
              </InputGroup>
              <Select
                items={periodItems}
                value={entry.timePeriod}
                onValueChange={(period: string | null) => period && updateEntry(entry.id, { timePeriod: period })}
              >
                <SelectTrigger className="w-[150px]" disabled={!premiumUser} title={hintWhenLocked}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {periodItems.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {spent !== undefined && (
              <div className="text-[11px] text-muted-foreground mt-2 ml-1">
                {t("keyTeamHelpers.modelMaxBudgetEditor.currentSpend")}
                {spent}
                {entry.budgetLimit !== null &&
                  t("keyTeamHelpers.modelMaxBudgetEditor.ofLimit", { limit: entry.budgetLimit })}
              </div>
            )}
          </div>
        );
      })}
      <Button variant="outline" size="sm" onClick={addEntry} disabled={!premiumUser} title={hintWhenLocked}>
        <Plus className="w-3 h-3" />
        {t("keyTeamHelpers.modelMaxBudgetEditor.addModelBudget")}
      </Button>
    </div>
  );
}

interface ModelMaxBudgetFieldProps extends ModelMaxBudgetEditorProps {
  hint: string;
  label?: string;
}

/** The editor with its label, so every form that offers it presents it the same way. */
export function ModelMaxBudgetField({
  hint,
  label = i18n.t("keyTeamHelpers.modelMaxBudgetEditor.fieldLabel"),
  ...editorProps
}: ModelMaxBudgetFieldProps) {
  return (
    <Field>
      <FieldLabel>
        <span title={hint}>{label}</span>
      </FieldLabel>
      <ModelMaxBudgetEditor {...editorProps} />
    </Field>
  );
}
