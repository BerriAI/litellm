import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import React from "react";

export type McpToolSearchSetting = boolean | null | undefined;

const OPTIONS = [
  { value: "inherit", label: "Not set" },
  { value: "enabled", label: "Enabled" },
  { value: "disabled", label: "Disabled" },
] as const;

const toOptionValue = (value: McpToolSearchSetting): string => {
  if (value === true) {
    return "enabled";
  }
  if (value === false) {
    return "disabled";
  }
  return "inherit";
};

const fromOptionValue = (value: string | null): McpToolSearchSetting => {
  if (value === "enabled") {
    return true;
  }
  if (value === "disabled") {
    return false;
  }
  return null;
};

interface MCPToolSearchSelectProps {
  id?: string;
  value: McpToolSearchSetting;
  onChange: (value: McpToolSearchSetting) => void;
}

export function MCPToolSearchSelect({ id, value, onChange }: MCPToolSearchSelectProps) {
  return (
    <Select items={OPTIONS} value={toOptionValue(value)} onValueChange={(v) => onChange(fromOptionValue(v))}>
      <SelectTrigger id={id} className="w-full">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {OPTIONS.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
