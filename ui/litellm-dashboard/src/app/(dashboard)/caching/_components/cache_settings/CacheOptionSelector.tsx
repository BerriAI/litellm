import React from "react";
import { Select } from "antd";

export interface CacheSelectOption {
  readonly value: string;
  readonly label: string;
}

interface CacheOptionSelectorProps {
  label: string;
  value: string;
  options: readonly CacheSelectOption[];
  descriptions: Readonly<Record<string, string>>;
  fallbackDescription: string;
  onChange: (value: string) => void;
}

const CacheOptionSelector: React.FC<CacheOptionSelectorProps> = ({
  label,
  value,
  options,
  descriptions,
  fallbackDescription,
  onChange,
}) => (
  <div className="space-y-2">
    <label className="text-sm font-medium text-gray-700">{label}</label>
    <Select
      value={value}
      onChange={(next) => onChange(next)}
      style={{ width: "100%" }}
      options={options.map((option) => ({ value: option.value, label: option.label }))}
    />
    <p className="text-xs text-gray-500">{descriptions[value] || fallbackDescription}</p>
  </div>
);

export default CacheOptionSelector;
