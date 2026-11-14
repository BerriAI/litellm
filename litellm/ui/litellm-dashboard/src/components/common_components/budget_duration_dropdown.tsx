import React from "react";
import type { ParseKeys, TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import i18n from "@/i18n/i18n";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export const NEVER_RESETS_BUDGET_DURATION = "none";

const DURATION_LABELS: Record<string, ParseKeys> = {
  [NEVER_RESETS_BUDGET_DURATION]: "budgetDurationDropdown.neverResets",
  "1h": "budgetDurationDropdown.periods.hourly",
  "24h": "budgetDurationDropdown.periods.daily",
  "7d": "budgetDurationDropdown.periods.weekly",
  "30d": "budgetDurationDropdown.periods.monthly",
  "12mo": "budgetDurationDropdown.periods.annual",
};

interface BudgetDurationDropdownProps {
  id?: string;
  value?: string | null;
  onChange?: (value: string | null) => void;
  className?: string;
  style?: React.CSSProperties;
  placeholder?: string;
  showNeverResets?: boolean;
}

const BudgetDurationDropdown: React.FC<BudgetDurationDropdownProps> = ({
  id,
  value,
  onChange,
  className = "",
  style = {},
  placeholder = i18n.t("budgetDurationDropdown.neverResets"),
  showNeverResets = false,
}) => {
  const { t } = useTranslation();
  const durationItems = Object.fromEntries(
    Object.entries(DURATION_LABELS).map(([duration, key]) => [duration, t(key)]),
  );
  return (
    <Select items={durationItems} value={value || null} onValueChange={onChange}>
      <SelectTrigger id={id} className={`w-full ${className}`} style={style}>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={null}>{placeholder}</SelectItem>
        {showNeverResets ? (
          <SelectItem value={NEVER_RESETS_BUDGET_DURATION}>{t("budgetDurationDropdown.neverResets")}</SelectItem>
        ) : null}
        <SelectItem value="1h">{t("budgetDurationDropdown.periods.hourly")}</SelectItem>
        <SelectItem value="24h">{t("budgetDurationDropdown.periods.daily")}</SelectItem>
        <SelectItem value="7d">{t("budgetDurationDropdown.periods.weekly")}</SelectItem>
        <SelectItem value="30d">{t("budgetDurationDropdown.periods.monthly")}</SelectItem>
        <SelectItem value="12mo">{t("budgetDurationDropdown.periods.annual")}</SelectItem>
      </SelectContent>
    </Select>
  );
};

export const getBudgetDurationLabel = (value: string | null | undefined, t: TFunction = i18n.t.bind(i18n)): string => {
  if (!value) return t("budgetDurationDropdown.notSet");

  const budgetDurationMap: Record<string, ParseKeys> = {
    "1h": "budgetDurationDropdown.periods.hourly",
    "24h": "budgetDurationDropdown.periods.daily",
    "7d": "budgetDurationDropdown.periods.weekly",
    "30d": "budgetDurationDropdown.periods.monthly",
    "12mo": "budgetDurationDropdown.periods.annual",
  };

  return Object.hasOwn(budgetDurationMap, value) ? t(budgetDurationMap[value]) : value;
};

export default BudgetDurationDropdown;
