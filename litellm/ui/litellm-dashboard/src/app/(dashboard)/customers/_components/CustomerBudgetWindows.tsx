import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import CustomerBudgetAmountInput from "./CustomerBudgetAmountInput";
import CustomerBudgetPeriodSelect from "./CustomerBudgetPeriodSelect";
import type { BudgetEditorFields } from "./customerBudgetForm";

type Windows = BudgetEditorFields["budgetWindows"];

export default function CustomerBudgetWindows({
  id,
  windows,
  onChange,
  disabled = false,
}: {
  id: string;
  windows: Windows;
  onChange: (windows: Windows) => void;
  disabled?: boolean;
}) {
  const { t } = useTranslation();
  const rows = windows.map(({ budget, maxBudget }, index) => {
    const period = budget.budget_duration;
    const inputId = `${id}-window-${index}`;
    const changeAmount = (value: string) =>
      onChange(windows.map((entry, i) => (i === index ? { ...entry, maxBudget: value } : entry)));
    return (
      <div key={index} className="space-y-2">
        <Label htmlFor={inputId} className="font-normal text-muted-foreground">
          {t("apps.additionalSpendLimit")}
        </Label>
        <div className="flex items-center gap-2">
          <CustomerBudgetAmountInput
            id={inputId}
            label={t("apps.additionalSpendLimit")}
            aria-describedby={`${id}-windows-hint`}
            value={maxBudget}
            onChange={changeAmount}
            disabled={disabled}
          >
            <CustomerBudgetPeriodSelect
              value={period}
              standardOnly
              disabled={disabled}
              unavailablePeriods={windows.filter((_, i) => i !== index).map((entry) => entry.budget.budget_duration)}
              onChange={(value) => {
                if (value === period || (value !== "1mo" && value !== "12mo")) return;
                onChange(
                  windows.map((entry, i) =>
                    i === index
                      ? { ...entry, budget: { ...entry.budget, budget_duration: value, reset_at: null } }
                      : entry,
                  ),
                );
              }}
            />
          </CustomerBudgetAmountInput>
          <Button
            type="button"
            variant="outline"
            disabled={disabled}
            onClick={() => onChange(windows.filter((_, i) => i !== index))}
          >
            {t("apps.removeWindow")}
          </Button>
        </div>
      </div>
    );
  });
  return (
    <>
      {windows.length > 0 && (
        <p id={`${id}-windows-hint`} className="text-xs text-muted-foreground">
          {t("apps.windowBudgetHint")}
        </p>
      )}
      {rows}
    </>
  );
}
