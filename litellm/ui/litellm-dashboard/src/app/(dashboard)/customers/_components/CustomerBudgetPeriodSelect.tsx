import { useId, useState } from "react";
import type { ParseKeys, TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { isValidBudgetDuration, normalizeBudgetDuration } from "@/utils/budgetUtils";

const periodLabels: Record<string, ParseKeys> = {
  "1mo": "customerPortal.createDialog.perMonth",
  "12mo": "customerPortal.createDialog.perYear",
  none: "customerPortal.budgetEditor.noReset",
  custom: "customerPortal.budgetEditor.customPeriod",
};
const compactPeriodLabels: Record<string, ParseKeys> = {
  "1mo": "customerPortal.createDialog.perMonthShort",
  "12mo": "customerPortal.createDialog.perYearShort",
};

function periodError(value: string, edited: boolean, unavailablePeriods: readonly string[], t: TFunction): string {
  if (!edited) return "";
  if (unavailablePeriods.includes(normalizeBudgetDuration(value)))
    return t("customerPortal.budgetEditor.duplicatePeriod");
  if (!isValidBudgetDuration(value)) return t("customerPortal.budgetEditor.invalidPeriod");
  return "";
}

export default function CustomerBudgetPeriodSelect({
  value,
  onChange,
  disabled,
  allowNoReset = true,
  unavailablePeriods = [],
  compact = false,
}: {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  allowNoReset?: boolean;
  unavailablePeriods?: readonly string[];
  compact?: boolean;
}) {
  const { t } = useTranslation();
  const errorId = useId();
  const [custom, setCustom] = useState(false);
  const customPeriod = custom || !["1mo", "12mo", ...(allowNoReset ? [""] : [])].includes(value);
  const period = customPeriod ? "custom" : value || "none";
  const unavailable = unavailablePeriods.map(normalizeBudgetDuration);
  const error = periodError(value, custom, unavailable, t);
  const changePeriod = (next: string | null) => {
    setCustom(next === "custom");
    if (next === "none") onChange("");
    if (next === "1mo" || next === "12mo") onChange(next);
  };
  return (
    <>
      {customPeriod && (
        <>
          <span className="pl-1.5 text-muted-foreground" aria-hidden="true">
            /
          </span>
          <Input
            ref={(input) => input?.setCustomValidity(error)}
            aria-label={t("customerPortal.budgetEditor.customPeriod")}
            aria-invalid={!!error}
            aria-describedby={error ? errorId : undefined}
            value={value}
            disabled={disabled}
            placeholder={t("customerPortal.budgetEditor.customPeriodPlaceholder")}
            className={`${compact ? "w-12 px-1.5 text-xs md:text-xs" : "w-20"} shrink-0 rounded-none border-0 shadow-none`}
            onChange={(event) => {
              setCustom(true);
              onChange(event.target.value);
            }}
          />
        </>
      )}
      <Select value={period} onValueChange={changePeriod} disabled={disabled}>
        <SelectTrigger
          aria-label={t("customerPortal.hierarchy.budgetDuration")}
          size={compact ? "sm" : "default"}
          className={`h-full shrink-0 rounded-l-none border-0 shadow-none ${compact ? "px-1.5 text-xs text-muted-foreground" : "border-l bg-muted/30 px-3"} ${compact && !customPeriod ? "[&>svg]:hidden" : ""}`}
        >
          <SelectValue className={customPeriod ? "sr-only" : undefined}>
            {t((compact && compactPeriodLabels[period]) || periodLabels[period])}
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          {Object.entries(periodLabels)
            .filter(([value]) => allowNoReset || value !== "none")
            .map(([value, label]) => (
              <SelectItem key={value} value={value} disabled={unavailable.includes(value)}>
                {t(label)}
              </SelectItem>
            ))}
        </SelectContent>
      </Select>
      {error && (
        <span
          id={errorId}
          role="alert"
          className="order-last max-w-64 basis-full whitespace-normal px-2 pb-1 text-xs text-destructive"
        >
          {error}
        </span>
      )}
    </>
  );
}
