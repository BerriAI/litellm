import { Button, InputNumber, Select } from "antd";
import React from "react";

export type ModelMaxBudgetConfig = {
  budget_limit: number;
  time_period: string;
};

export type ModelMaxBudgetValue = Record<string, ModelMaxBudgetConfig>;

export type ModelMaxBudgetEntry = {
  model: string;
  budget_limit: number | null;
  time_period: string;
};

const TIME_PERIOD_OPTIONS = [
  { value: "1h", label: "Hourly" },
  { value: "24h", label: "Daily" },
  { value: "1d", label: "Daily (1d)" },
  { value: "7d", label: "Weekly" },
  { value: "30d", label: "Monthly" },
];

const TIME_PERIOD_LABELS: Record<string, string> = {
  "1h": "hour",
  "24h": "day",
  "1d": "day",
  "7d": "week",
  "30d": "month",
};

export function normalizeModelMaxBudgetConfig(
  config: ModelMaxBudgetConfig | Record<string, unknown>,
): ModelMaxBudgetConfig {
  const raw = config as Record<string, unknown>;
  return {
    budget_limit: Number(raw.budget_limit ?? raw.max_budget ?? 0),
    time_period: String(raw.time_period ?? raw.budget_duration ?? "1d"),
  };
}

export function formatModelMaxBudgetTimePeriod(timePeriod: string): string {
  return TIME_PERIOD_LABELS[timePeriod] ?? timePeriod;
}

function normalizeConfig(config: ModelMaxBudgetConfig | Record<string, unknown>): ModelMaxBudgetConfig {
  return normalizeModelMaxBudgetConfig(config);
}

export function modelMaxBudgetEntriesFromValue(
  value: ModelMaxBudgetValue | null | undefined,
): ModelMaxBudgetEntry[] {
  if (!value) {
    return [];
  }
  return Object.entries(value).map(([model, config]) => {
    const normalized = normalizeConfig(config);
    return {
      model,
      budget_limit: normalized.budget_limit,
      time_period: normalized.time_period,
    };
  });
}

export function modelMaxBudgetValueFromEntries(entries: ModelMaxBudgetEntry[]): ModelMaxBudgetValue | null {
  const filtered = entries.filter((entry) => entry.model && entry.budget_limit !== null && entry.budget_limit > 0);
  if (filtered.length === 0) {
    return null;
  }
  return Object.fromEntries(
    filtered.map((entry) => [
      entry.model,
      {
        budget_limit: entry.budget_limit as number,
        time_period: entry.time_period,
      },
    ]),
  );
}

interface ModelMaxBudgetEditorProps {
  value?: ModelMaxBudgetValue | null;
  onChange?: (value: ModelMaxBudgetValue | null) => void;
  modelOptions: string[];
  disabled?: boolean;
}

export function ModelMaxBudgetEditor({ value, onChange, modelOptions, disabled = false }: ModelMaxBudgetEditorProps) {
  const entries = modelMaxBudgetEntriesFromValue(value);

  const updateEntries = (nextEntries: ModelMaxBudgetEntry[]) => {
    if (disabled) {
      return;
    }
    onChange?.(modelMaxBudgetValueFromEntries(nextEntries));
  };

  const addEntry = () => {
    const unusedModel =
      modelOptions.find((model) => !entries.some((entry) => entry.model === model)) ?? modelOptions[0] ?? "";
    updateEntries([...entries, { model: unusedModel, budget_limit: 1, time_period: "1d" }]);
  };

  const removeEntry = (index: number) => {
    updateEntries(entries.filter((_, entryIndex) => entryIndex !== index));
  };

  const updateEntry = (index: number, field: keyof ModelMaxBudgetEntry, fieldValue: string | number | null) => {
    updateEntries(
      entries.map((entry, entryIndex) => (entryIndex === index ? { ...entry, [field]: fieldValue } : entry)),
    );
  };

  return (
    <div>
      {entries.map((entry, index) => (
        <div key={`${entry.model}-${index}`} style={{ display: "flex", gap: 8, marginBottom: 12, alignItems: "center" }}>
          <Select
            showSearch
            value={entry.model || undefined}
            onChange={(selectedModel) => updateEntry(index, "model", selectedModel)}
            style={{ width: 220 }}
            placeholder="Select model"
            options={modelOptions.map((model) => ({ value: model, label: model }))}
            disabled={disabled}
          />
          <InputNumber
            step={0.01}
            min={0}
            precision={2}
            value={entry.budget_limit ?? undefined}
            onChange={(nextValue) => updateEntry(index, "budget_limit", nextValue ?? null)}
            placeholder="Max spend ($)"
            style={{ width: 160 }}
            prefix="$"
            disabled={disabled}
          />
          <Select
            value={entry.time_period}
            onChange={(nextPeriod) => updateEntry(index, "time_period", nextPeriod)}
            style={{ width: 140 }}
            options={TIME_PERIOD_OPTIONS}
            disabled={disabled}
          />
          <Button
            type="text"
            danger
            size="small"
            onClick={() => removeEntry(index)}
            style={{ padding: "0 4px" }}
            disabled={disabled}
          >
            ✕
          </Button>
        </div>
      ))}
      <Button
        size="small"
        onClick={addEntry}
        disabled={disabled || modelOptions.length === 0}
        data-testid="add-model-budget-button"
      >
        + Add Model Budget
      </Button>
    </div>
  );
}
