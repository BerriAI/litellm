import React from "react";
import { Select } from "antd";

const { Option } = Select;

export const NEVER_RESETS_BUDGET_DURATION = "none";

interface BudgetDurationDropdownProps {
  value?: string | null;
  onChange?: (value: string | undefined) => void;
  className?: string;
  style?: React.CSSProperties;
  placeholder?: string;
  showNeverResets?: boolean;
}

const BudgetDurationDropdown: React.FC<BudgetDurationDropdownProps> = ({
  value,
  onChange,
  className = "",
  style = {},
  placeholder = "n/a",
  showNeverResets = false,
}) => {
  return (
    <Select
      style={{ width: "100%", ...style }}
      value={value || undefined}
      onChange={onChange}
      className={className}
      placeholder={placeholder}
      allowClear
    >
      {showNeverResets ? <Option value={NEVER_RESETS_BUDGET_DURATION}>Never resets</Option> : null}
      <Option value="1h">hourly</Option>
      <Option value="24h">daily</Option>
      <Option value="7d">weekly</Option>
      <Option value="30d">monthly</Option>
    </Select>
  );
};

export const getBudgetDurationLabel = (value: string | null | undefined): string => {
  if (!value) return "Not set";

  const budgetDurationMap: Record<string, string> = {
    "1h": "hourly",
    "24h": "daily",
    "7d": "weekly",
    "30d": "monthly",
  };

  return budgetDurationMap[value] || value;
};

export default BudgetDurationDropdown;
