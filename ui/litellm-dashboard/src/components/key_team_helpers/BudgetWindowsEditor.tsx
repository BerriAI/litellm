import { Button, InputNumber, Select } from "antd";
import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";

export interface BudgetWindowEntry {
  budget_duration: string;
  max_budget: number | null;
}

const getBudgetWindowOptions = (t: TFunction) => [
  {
    value: "1h",
    label: t("keyTeamHelpers.budgetWindowsEditor.hourly"),
    resetHint: t("keyTeamHelpers.budgetWindowsEditor.resetsEveryHour"),
  },
  {
    value: "24h",
    label: t("keyTeamHelpers.budgetWindowsEditor.daily"),
    resetHint: t("keyTeamHelpers.budgetWindowsEditor.resetsDailyMidnight"),
  },
  {
    value: "7d",
    label: t("keyTeamHelpers.budgetWindowsEditor.weekly"),
    resetHint: t("keyTeamHelpers.budgetWindowsEditor.resetsWeeklySunday"),
  },
  {
    value: "30d",
    label: t("keyTeamHelpers.budgetWindowsEditor.monthly"),
    resetHint: t("keyTeamHelpers.budgetWindowsEditor.resetsMonthly"),
  },
];

interface BudgetWindowsEditorProps {
  value: BudgetWindowEntry[];
  onChange: (v: BudgetWindowEntry[]) => void;
}

export function BudgetWindowsEditor({ value, onChange }: BudgetWindowsEditorProps) {
  const { t } = useTranslation();
  const budgetWindowOptions = useMemo(() => getBudgetWindowOptions(t), [t]);

  const addWindow = () => {
    onChange([...value, { budget_duration: "24h", max_budget: null }]);
  };

  const removeWindow = (idx: number) => {
    onChange(value.filter((_, i) => i !== idx));
  };

  const updateWindow = (idx: number, field: keyof BudgetWindowEntry, fieldValue: string | number | null) => {
    const updated = value.map((w, i) => (i === idx ? { ...w, [field]: fieldValue } : w));
    onChange(updated);
  };

  return (
    <div>
      {value.map((window, idx) => {
        const hint = budgetWindowOptions.find((o) => o.value === window.budget_duration)?.resetHint;
        return (
          <div key={idx} style={{ marginBottom: 12 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <Select
                value={window.budget_duration}
                onChange={(v) => updateWindow(idx, "budget_duration", v)}
                style={{ width: 130 }}
                options={budgetWindowOptions.map((o) => ({ value: o.value, label: o.label }))}
              />
              <InputNumber
                step={0.01}
                min={0}
                precision={2}
                value={window.max_budget ?? undefined}
                onChange={(v) => updateWindow(idx, "max_budget", v ?? null)}
                placeholder={t("keyTeamHelpers.budgetWindowsEditor.maxSpendPlaceholder")}
                style={{ width: 160 }}
                prefix="$"
              />
              <Button type="text" danger size="small" onClick={() => removeWindow(idx)} style={{ padding: "0 4px" }}>
                ✕
              </Button>
            </div>
            {hint && <div style={{ fontSize: 11, color: "#888", marginTop: 3, marginLeft: 2 }}>↻ {hint}</div>}
          </div>
        );
      })}
      <Button
        size="small"
        onClick={(e) => {
          e.preventDefault();
          addWindow();
        }}
      >
        {t("keyTeamHelpers.budgetWindowsEditor.addBudgetWindow")}
      </Button>
    </div>
  );
}
