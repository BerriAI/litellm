import React from "react";
import { Select } from "antd";

const { Option } = Select;

interface BudgetDurationDropdownProps {
  value?: string | null;
  onChange: (value: string) => void;
  className?: string;
  style?: React.CSSProperties;
}

const BudgetDurationDropdown: React.FC<BudgetDurationDropdownProps> = ({
  value,
  onChange,
  className = "",
  style = {}
}) => {
  return (
    <Select
      style={{ width: '100%', ...style }}
      value={value || undefined}
      onChange={onChange}
      className={className}
      placeholder="n/a"
    >
      <Option value="1d">daily</Option>
      <Option value="1w">weekly</Option>
      <Option value="1mo">monthly</Option>
    </Select>
  );
};

export const getBudgetDurationLabel = (value: string | null | undefined): string => {
  if (!value) return "Not set";
  
  const budgetDurationMap: Record<string, string> = {
    "1d": "daily",
    "1w": "weekly",
    "1mo": "monthly"
  };
  
  return budgetDurationMap[value] || value;
};

export default BudgetDurationDropdown; 