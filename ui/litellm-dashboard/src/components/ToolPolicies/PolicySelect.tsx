"use client";

import React from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/cva.config";

export const INPUT_POLICY_OPTIONS = [
  { value: "untrusted", label: "untrusted", dot: "bg-amber-500" },
  { value: "trusted", label: "trusted", dot: "bg-green-500" },
  { value: "blocked", label: "blocked", dot: "bg-red-500" },
] as const;

export const OUTPUT_POLICY_OPTIONS = [
  { value: "untrusted", label: "untrusted", dot: "bg-amber-500" },
  { value: "trusted", label: "trusted", dot: "bg-green-500" },
] as const;

export const POLICY_OPTIONS = INPUT_POLICY_OPTIONS;

export const policyStyle = (p: string) => INPUT_POLICY_OPTIONS.find((o) => o.value === p) ?? INPUT_POLICY_OPTIONS[0];

export interface PolicySelectProps {
  value: string;
  toolName: string;
  saving: boolean;
  onChange: (toolName: string, policy: string) => void;
  policyType?: "input" | "output";
  size?: "small" | "middle";
  minWidth?: number;
  stopPropagation?: boolean;
}

export const PolicySelect: React.FC<PolicySelectProps> = ({
  value,
  toolName,
  saving,
  onChange,
  policyType = "input",
  size = "small",
  stopPropagation = true,
}) => {
  const options = policyType === "output" ? OUTPUT_POLICY_OPTIONS : INPUT_POLICY_OPTIONS;
  const selected = policyStyle(value);
  return (
    <Select value={value} disabled={saving} onValueChange={(v: string | null) => v !== null && onChange(toolName, v)}>
      <SelectTrigger
        size={size === "small" ? "sm" : "default"}
        className="w-auto min-w-28"
        onClick={(e) => stopPropagation && e.stopPropagation()}
      >
        <span className={cn("size-2 shrink-0 rounded-full", selected.dot)} />
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((o) => (
          <SelectItem key={o.value} value={o.value}>
            <span className="inline-flex items-center gap-1.5">
              <span className={cn("size-2 shrink-0 rounded-full", o.dot)} />
              {o.label}
            </span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};
