import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import React from "react";

const TierRowSelect: React.FC<{
  label: string;
  options: { value: string; label: string }[];
  value: string | null;
  onValueChange: (rowId: string) => void;
  placeholder?: string;
}> = ({ label, options, value, onValueChange, placeholder }) => (
  <Select items={options} value={value} onValueChange={(rowId: string | null) => rowId && onValueChange(rowId)}>
    <SelectTrigger aria-label={label} className="w-full">
      <SelectValue placeholder={placeholder} />
    </SelectTrigger>
    <SelectContent>
      {options.map((option) => (
        <SelectItem key={option.value} value={option.value}>
          {option.label}
        </SelectItem>
      ))}
    </SelectContent>
  </Select>
);

export default TierRowSelect;
