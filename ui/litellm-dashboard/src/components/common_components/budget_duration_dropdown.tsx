import { useLanguage } from "@/contexts/LanguageContext";
import { englishTranslate, type Translate } from "@/i18n/language";
import React from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export const NEVER_RESETS_BUDGET_DURATION = "none";

const DURATION_LABELS: Record<string, string> = {
  [NEVER_RESETS_BUDGET_DURATION]: "Never resets",
  "1h": "hourly",
  "24h": "daily",
  "7d": "weekly",
  "30d": "monthly",
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
  placeholder,
  showNeverResets = false,
}) => {
  const { t } = useLanguage();
  const placeholderText = placeholder ?? t("Never resets");
  // 标签本地化，选中值和提交周期仍使用原始协议值
  const items = Object.fromEntries(Object.entries(DURATION_LABELS).map(([duration, label]) => [duration, t(label)]));
  return (
    <Select items={items} value={value || null} onValueChange={onChange}>
      <SelectTrigger id={id} className={`w-full ${className}`} style={style}>
        <SelectValue placeholder={placeholderText} />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={null}>{placeholderText}</SelectItem>
        {showNeverResets ? <SelectItem value={NEVER_RESETS_BUDGET_DURATION}>{t("Never resets")}</SelectItem> : null}
        <SelectItem value="1h">{t("hourly")}</SelectItem>
        <SelectItem value="24h">{t("daily")}</SelectItem>
        <SelectItem value="7d">{t("weekly")}</SelectItem>
        <SelectItem value="30d">{t("monthly")}</SelectItem>
      </SelectContent>
    </Select>
  );
};

export const getBudgetDurationLabel = (value: string | null | undefined, t: Translate = englishTranslate): string => {
  if (!value) return t("Not set");

  const budgetDurationMap: Record<string, string> = {
    "1h": "hourly",
    "24h": "daily",
    "7d": "weekly",
    "30d": "monthly",
  };

  return Object.hasOwn(budgetDurationMap, value) ? t(budgetDurationMap[value]) : value;
};

export default BudgetDurationDropdown;
