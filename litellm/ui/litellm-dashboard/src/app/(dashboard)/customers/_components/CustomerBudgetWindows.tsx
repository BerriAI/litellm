import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import CustomerBudgetAmountInput from "./CustomerBudgetAmountInput";
import CustomerBudgetPeriodSelect from "./CustomerBudgetPeriodSelect";
import type { BudgetEditorFields } from "./customerBudgetForm";
import { isCalendarBudgetDuration } from "@/lib/forms/budgetWindows";

type Windows = BudgetEditorFields["budgetWindows"];

export default function CustomerBudgetWindows({
  id,
  windows,
  onChange,
  removable = false,
  disabled = false,
}: {
  id: string;
  windows: Windows;
  onChange: (windows: Windows) => void;
  removable?: boolean;
  disabled?: boolean;
}) {
  const { t } = useTranslation();
  const rows = windows.map(({ budget, maxBudget }, index) => {
    const period = budget.budget_duration;
    if (!removable && isCalendarBudgetDuration(period)) return null;
    const inputId = `${id}-window-${index}`;
    const changeAmount = (value: string) =>
      onChange(windows.map((entry, i) => (i === index ? { ...entry, maxBudget: value } : entry)));
    return (
      <div key={index} className="space-y-2">
        <Label htmlFor={inputId} className="font-normal text-muted-foreground">
          {t(removable ? "apps.additionalSpendLimit" : "apps.windowBudget", { period })}
        </Label>
        <div className={removable ? "flex items-center gap-2" : undefined}>
          {removable ? (
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
                allowNoReset={false}
                disabled={disabled}
                onChange={(value) =>
                  onChange(
                    windows.map((entry, i) =>
                      i === index ? { ...entry, budget: { ...entry.budget, budget_duration: value } } : entry,
                    ),
                  )
                }
              />
            </CustomerBudgetAmountInput>
          ) : (
            <Input
              id={inputId}
              type="number"
              min={0}
              step="any"
              value={maxBudget}
              disabled={disabled}
              placeholder={t("customerPortal.createDialog.noLimit")}
              onChange={(event) => changeAmount(event.target.value)}
            />
          )}
          {removable && (
            <Button
              type="button"
              variant="outline"
              disabled={disabled}
              onClick={() => onChange(windows.filter((_, i) => i !== index))}
            >
              {t("apps.removeWindow")}
            </Button>
          )}
        </div>
      </div>
    );
  });
  return (
    <>
      {removable && windows.length > 0 && (
        <p id={`${id}-windows-hint`} className="text-xs text-muted-foreground">
          {t("apps.windowBudgetHint")}
        </p>
      )}
      {rows}
    </>
  );
}
