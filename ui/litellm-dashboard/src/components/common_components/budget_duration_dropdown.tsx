import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

interface BudgetDurationDropdownProps {
  value?: string | null;
  onChange?: (value: string | null) => void;
  className?: string;
  style?: React.CSSProperties;
}

const BudgetDurationDropdown: React.FC<BudgetDurationDropdownProps> = ({
  value,
  onChange,
  className = "",
  style = {},
}) => {
  return (
    <div style={{ width: "100%", ...style }} className={cn(className)}>
      <Select
        value={value ?? ""}
        onValueChange={(v) => onChange?.(v || null)}
      >
        <SelectTrigger>
          <SelectValue placeholder="n/a" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="1h">hourly</SelectItem>
          <SelectItem value="24h">daily</SelectItem>
          <SelectItem value="7d">weekly</SelectItem>
          <SelectItem value="30d">monthly</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
};

export const getBudgetDurationLabel = (
  value: string | null | undefined,
): string => {
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
