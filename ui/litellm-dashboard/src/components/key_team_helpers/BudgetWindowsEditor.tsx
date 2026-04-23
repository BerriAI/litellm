import React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { X } from "lucide-react";

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

  const updateWindow = (
    idx: number,
    field: keyof BudgetWindowEntry,
    fieldValue: string | number | null,
  ) => {
    const updated = value.map((w, i) => (i === idx ? { ...w, [field]: fieldValue } : w));
    onChange(updated);
  };

  return (
    <div>
      {value.map((window, idx) => {
        const hint = BUDGET_WINDOW_OPTIONS.find(
          (o) => o.value === window.budget_duration,
        )?.resetHint;
        return (
          <div key={idx} className="mb-3">
            <div className="flex gap-2 items-center">
              <Select
                value={window.budget_duration}
                onValueChange={(v) => updateWindow(idx, "budget_duration", v)}
              >
                <SelectTrigger className="w-[130px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {BUDGET_WINDOW_OPTIONS.map((o) => (
                    <SelectItem key={o.value} value={o.value}>
                      {o.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <div className="relative w-[160px]">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm pointer-events-none">
                  $
                </span>
                <Input
                  type="number"
                  step={0.01}
                  min={0}
                  value={window.max_budget ?? ""}
                  onChange={(e) => {
                    const v = e.target.value;
                    updateWindow(idx, "max_budget", v === "" ? null : Number(v));
                  }}
                  placeholder="Max spend"
                  className="pl-6"
                />
              </div>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => removeWindow(idx)}
                className="text-destructive hover:text-destructive"
                aria-label="Remove budget window"
              >
                <X size={14} />
              </Button>
            </div>
            {hint && (
              <div className="text-[11px] text-muted-foreground mt-1 ml-0.5">
                ↻ {hint}
              </div>
            )}
          </div>
        );
      })}
      <Button
        type="button"
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
