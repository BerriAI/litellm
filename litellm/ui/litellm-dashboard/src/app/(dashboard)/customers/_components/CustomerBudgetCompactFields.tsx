import { type BudgetEditorFields, budgetPeriodAmount, updateBudgetPeriod } from "./customerBudgetForm";
import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import CustomerBudgetAmountInput from "./CustomerBudgetAmountInput";
import { Label } from "@/components/ui/label";
import { CustomerBudgetAdvancedFields } from "./CustomerBudgetFields";
import type { Budget } from "./customerBudgetForm";
import CustomerBudgetWindows from "./CustomerBudgetWindows";
import { CustomerRateLimitControls } from "./CustomerRateLimits";

export default function CustomerBudgetCompactFields({
  fields,
  original,
  onChange,
  disabled,
  id,
}: {
  fields: BudgetEditorFields;
  original?: Budget | null;
  onChange: (fields: BudgetEditorFields) => void;
  disabled: boolean;
  id: string;
}) {
  const { t } = useTranslation();
  const periods = ["1mo", "12mo"] as const;
  const hasOtherSpendLimit =
    original?.max_budget != null && original.budget_duration !== "1mo" && original.budget_duration !== "12mo";
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {periods.map((period) => (
          <div key={period} className="min-w-0 space-y-2">
            <Label htmlFor={`${id}-${period}`} className="font-normal text-muted-foreground">
              {t(period === "1mo" ? "apps.monthlyBudget" : "apps.yearlyBudget")}
            </Label>
            <CustomerBudgetAmountInput
              id={`${id}-${period}`}
              label={t(period === "1mo" ? "apps.monthlyBudget" : "apps.yearlyBudget")}
              disabled={disabled}
              value={budgetPeriodAmount(fields, period)}
              onChange={(value) => onChange(updateBudgetPeriod(fields, period, value))}
            />
          </div>
        ))}
        <CustomerRateLimitControls
          id={id}
          tpm={fields.tpm}
          rpm={fields.rpm}
          disabled={disabled}
          onChange={(key, value) => onChange({ ...fields, [key]: value })}
        />
      </div>
      {hasOtherSpendLimit && (
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor={`${id}-other-spend`} className="font-normal text-muted-foreground">
              {t("apps.additionalSpendLimit")}
            </Label>
            <Input
              id={`${id}-other-spend`}
              type="number"
              min={0}
              step="any"
              disabled={disabled}
              value={fields.maxBudget}
              onChange={(event) => onChange({ ...fields, maxBudget: event.target.value })}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor={`${id}-duration`} className="font-normal text-muted-foreground">
              {t("apps.customResetPeriod")}
            </Label>
            <Input
              id={`${id}-duration`}
              disabled={disabled}
              value={fields.budgetDuration}
              placeholder={t("apps.noReset")}
              onChange={(event) => onChange({ ...fields, budgetDuration: event.target.value })}
            />
          </div>
        </div>
      )}
      <details className="text-sm">
        <summary className="cursor-pointer text-muted-foreground">{t("apps.advancedLimits")}</summary>
        <div className="mt-4 space-y-4">
          <CustomerBudgetWindows
            id={id}
            windows={fields.budgetWindows}
            onChange={(budgetWindows) => onChange({ ...fields, budgetWindows })}
          />
          <CustomerBudgetAdvancedFields
            idPrefix={id}
            fields={fields}
            disabled={disabled}
            onChange={(key, value) => onChange({ ...fields, [key]: value })}
          />
        </div>
      </details>
    </div>
  );
}
