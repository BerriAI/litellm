import React from "react";
import { Select } from "antd";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";

const { Option } = Select;

interface BudgetDurationDropdownProps {
  value?: string | null;
  onChange?: (value: string | undefined) => void;
  className?: string;
  style?: React.CSSProperties;
  placeholder?: string;
}

const BudgetDurationDropdown: React.FC<BudgetDurationDropdownProps> = ({
  value,
  onChange,
  className = "",
  style = {},
  placeholder = "n/a",
}) => {
  const { t } = useTranslation("gateway");
  return (
    <Select
      style={{ width: "100%", ...style }}
      value={value || undefined}
      onChange={onChange}
      className={className}
      placeholder={placeholder}
      allowClear
    >
      <Option value="1h">{t("budgets.duration.hourly")}</Option>
      <Option value="24h">{t("budgets.duration.daily")}</Option>
      <Option value="7d">{t("budgets.duration.weekly")}</Option>
      <Option value="30d">{t("budgets.duration.monthly")}</Option>
    </Select>
  );
};

export const getBudgetDurationLabel = (value: string | null | undefined, t?: TFunction<"gateway">): string => {
  if (!value) return t ? t("budgets.duration.notSet") : "Not set";

  const budgetDurationMap: Record<string, string> = {
    "1h": t ? t("budgets.duration.hourly") : "hourly",
    "24h": t ? t("budgets.duration.daily") : "daily",
    "7d": t ? t("budgets.duration.weekly") : "weekly",
    "30d": t ? t("budgets.duration.monthly") : "monthly",
  };

  return budgetDurationMap[value] || value;
};

export default BudgetDurationDropdown;
