"use client";

import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

export const INPUT_POLICY_OPTIONS = [
  {
    value: "untrusted",
    label: "untrusted",
    className: "bg-amber-100 text-amber-800 border-amber-300",
    dotClass: "bg-amber-800",
  },
  {
    value: "trusted",
    label: "trusted",
    className: "bg-emerald-100 text-emerald-800 border-emerald-300",
    dotClass: "bg-emerald-800",
  },
  {
    value: "blocked",
    label: "blocked",
    className: "bg-red-100 text-red-800 border-red-300",
    dotClass: "bg-red-800",
  },
] as const;

export const OUTPUT_POLICY_OPTIONS = [
  INPUT_POLICY_OPTIONS[0],
  INPUT_POLICY_OPTIONS[1],
] as const;

export const POLICY_OPTIONS = INPUT_POLICY_OPTIONS;

export const policyStyle = (p: string) =>
  INPUT_POLICY_OPTIONS.find((o) => o.value === p) ?? INPUT_POLICY_OPTIONS[0];

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

// Color pill around the shadcn Select trigger. This is the one place in
// phase 1 where we reach for named Tailwind colors — the policy-badge
// palette (amber/emerald/red) is purposely non-semantic because it
// encodes a categorical (untrusted / trusted / blocked) not a theme state.
// Recorded in DEVIATIONS.md if we decide to force semantic-token
// conformance later.
export const PolicySelect: React.FC<PolicySelectProps> = ({
  value,
  toolName,
  saving,
  onChange,
  policyType = "input",
  size = "small",
  minWidth = 110,
  stopPropagation = true,
}) => {
  const options =
    policyType === "output" ? OUTPUT_POLICY_OPTIONS : INPUT_POLICY_OPTIONS;
  const style = policyStyle(value);
  const triggerHeight = size === "small" ? "h-7" : "h-8";
  const fontSize = size === "small" ? "text-[11px]" : "text-xs";
  return (
    <div
      onClick={(e) => stopPropagation && e.stopPropagation()}
      style={{ minWidth }}
    >
      <Select
        value={value}
        disabled={saving}
        onValueChange={(v) => onChange(toolName, v)}
      >
        <SelectTrigger
          className={cn(
            "font-medium rounded-full border",
            style.className,
            triggerHeight,
            fontSize,
          )}
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent className="w-auto">
          {options.map((o) => (
            <SelectItem key={o.value} value={o.value}>
              <span className={cn("inline-flex items-center gap-1.5", fontSize)}>
                <span
                  className={cn("w-2 h-2 rounded-full inline-block", o.dotClass)}
                />
                {o.label}
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};
