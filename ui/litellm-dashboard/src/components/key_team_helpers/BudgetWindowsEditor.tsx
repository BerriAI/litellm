import { Button } from "@/components/ui/button";
import { InputGroup, InputGroupAddon, InputGroupInput, InputGroupText } from "@/components/ui/input-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import React from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";

export interface BudgetWindowEntry {
  budget_duration: string;
  max_budget: number | null;
}

export const BUDGET_WINDOW_OPTIONS: readonly { value: string; labelKey: ParseKeys; resetHintKey: ParseKeys }[] = [
  {
    value: "1h",
    labelKey: "keyTeamHelpers.budgetWindowsEditor.options.hourly.label",
    resetHintKey: "keyTeamHelpers.budgetWindowsEditor.options.hourly.resetHint",
  },
  {
    value: "24h",
    labelKey: "keyTeamHelpers.budgetWindowsEditor.options.daily.label",
    resetHintKey: "keyTeamHelpers.budgetWindowsEditor.options.daily.resetHint",
  },
  {
    value: "7d",
    labelKey: "keyTeamHelpers.budgetWindowsEditor.options.weekly.label",
    resetHintKey: "keyTeamHelpers.budgetWindowsEditor.options.weekly.resetHint",
  },
  {
    value: "30d",
    labelKey: "keyTeamHelpers.budgetWindowsEditor.options.monthly.label",
    resetHintKey: "keyTeamHelpers.budgetWindowsEditor.options.monthly.resetHint",
  },
];

interface BudgetWindowsEditorProps {
  value: BudgetWindowEntry[];
  onChange: (v: BudgetWindowEntry[]) => void;
}

export function BudgetWindowsEditor({ value, onChange }: BudgetWindowsEditorProps) {
  const { t } = useTranslation();
  const windowItems = BUDGET_WINDOW_OPTIONS.map(({ value, labelKey }) => ({ value, label: t(labelKey) }));
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
        const hintKey = BUDGET_WINDOW_OPTIONS.find((o) => o.value === window.budget_duration)?.resetHintKey;
        return (
          <div key={idx} style={{ marginBottom: 12 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <Select
                items={windowItems}
                value={window.budget_duration}
                onValueChange={(v: string | null) => v && updateWindow(idx, "budget_duration", v)}
              >
                <SelectTrigger className="w-[130px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {windowItems.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <InputGroup className="w-40">
                <InputGroupAddon>
                  <InputGroupText>$</InputGroupText>
                </InputGroupAddon>
                <InputGroupInput
                  type="number"
                  step={0.01}
                  min={0}
                  value={window.max_budget ?? ""}
                  onChange={(event) => {
                    const typed = event.target.valueAsNumber;
                    updateWindow(idx, "max_budget", Number.isNaN(typed) ? null : typed);
                  }}
                  onBlur={(event) => {
                    const typed = event.target.valueAsNumber;
                    if (!Number.isNaN(typed)) {
                      updateWindow(idx, "max_budget", Number(typed.toFixed(2)));
                    }
                  }}
                  placeholder={t("keyTeamHelpers.budgetWindowsEditor.maxSpend")}
                />
              </InputGroup>
              <Button
                variant="ghost"
                size="sm"
                className="px-1 text-destructive hover:text-destructive/80"
                onClick={() => removeWindow(idx)}
              >
                ✕
              </Button>
            </div>
            {hintKey && <div style={{ fontSize: 11, color: "#888", marginTop: 3, marginLeft: 2 }}>↻ {t(hintKey)}</div>}
          </div>
        );
      })}
      <Button
        variant="outline"
        size="sm"
        onClick={(e) => {
          e.preventDefault();
          addWindow();
        }}
      >
        {t("keyTeamHelpers.budgetWindowsEditor.addWindow")}
      </Button>
    </div>
  );
}
