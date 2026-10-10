import { Button } from "@/components/ui/button";
import { InputGroup, InputGroupAddon, InputGroupInput, InputGroupText } from "@/components/ui/input-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import React from "react";

export interface BudgetWindowEntry {
  budget_duration: string;
  max_budget: number | null;
}

export const BUDGET_WINDOW_OPTIONS = [
  { value: "1h", label: "Hourly", resetHint: "Resets every hour" },
  { value: "24h", label: "Daily", resetHint: "Resets daily at midnight UTC" },
  { value: "7d", label: "Weekly", resetHint: "Resets every Sunday at midnight UTC" },
  { value: "30d", label: "Monthly", resetHint: "Resets on the 1st of every month at midnight UTC" },
];

interface BudgetWindowsEditorProps {
  value: BudgetWindowEntry[];
  onChange: (v: BudgetWindowEntry[]) => void;
}

export function BudgetWindowsEditor({ value, onChange }: BudgetWindowsEditorProps) {
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
        const hint = BUDGET_WINDOW_OPTIONS.find((o) => o.value === window.budget_duration)?.resetHint;
        return (
          <div key={idx} style={{ marginBottom: 12 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <Select
                items={BUDGET_WINDOW_OPTIONS}
                value={window.budget_duration}
                onValueChange={(v: string | null) => v && updateWindow(idx, "budget_duration", v)}
              >
                <SelectTrigger className="w-[130px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {BUDGET_WINDOW_OPTIONS.map((option) => (
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
                  placeholder="Max spend ($)"
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
            {hint && <div style={{ fontSize: 11, color: "#888", marginTop: 3, marginLeft: 2 }}>↻ {hint}</div>}
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
        + Add Budget Window
      </Button>
    </div>
  );
}
