"use client";

import React from "react";
import { Select } from "antd";

export const INPUT_POLICY_OPTIONS = [
  { value: "untrusted", label: "untrusted", color: "#92400e", bg: "#fef3c7", border: "#fcd34d" },
  { value: "trusted", label: "trusted", color: "#065f46", bg: "#d1fae5", border: "#6ee7b7" },
  { value: "blocked", label: "blocked", color: "#991b1b", bg: "#fee2e2", border: "#fca5a5" },
] as const;

export const OUTPUT_POLICY_OPTIONS = [
  { value: "untrusted", label: "untrusted", color: "#92400e", bg: "#fef3c7", border: "#fcd34d" },
  { value: "trusted", label: "trusted", color: "#065f46", bg: "#d1fae5", border: "#6ee7b7" },
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
  const options = policyType === "output" ? OUTPUT_POLICY_OPTIONS : INPUT_POLICY_OPTIONS;
  const style = policyStyle(value);
  return (
    <Select
      size={size}
      value={value}
      disabled={saving}
      loading={saving}
      onChange={(v) => onChange(toolName, v)}
      onClick={(e) => stopPropagation && e.stopPropagation()}
      style={{
        minWidth,
        fontWeight: 500,
        backgroundColor: style.bg,
        borderColor: style.border,
        color: style.color,
        borderRadius: 999,
        fontSize: size === "small" ? 11 : 12,
      }}
      popupMatchSelectWidth={false}
      options={options.map((o) => ({
        value: o.value,
        label: (
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              fontSize: 12,
              fontWeight: 500,
              color: o.color,
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                backgroundColor: o.color,
                display: "inline-block",
                flexShrink: 0,
              }}
            />
            {o.label}
          </span>
        ),
      }))}
    />
  );
};
