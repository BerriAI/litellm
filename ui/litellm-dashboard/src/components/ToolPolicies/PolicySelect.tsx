"use client";

import React from "react";
import { Select } from "antd";

// DB policy values: "trusted" | "untrusted" | "dual_llm" | "blocked" — we expose all except dual_llm in the Policy dropdown
export const POLICY_OPTIONS = [
  { value: "trusted", label: "trusted", color: "#065f46", bg: "#d1fae5", border: "#6ee7b7" },
  { value: "untrusted", label: "untrusted", color: "#92400e", bg: "#fef3c7", border: "#fcd34d" },
  { value: "blocked", label: "blocked", color: "#991b1b", bg: "#fee2e2", border: "#fca5a5" },
] as const;

export const policyStyle = (p: string) =>
  POLICY_OPTIONS.find((o) => o.value === p) ?? POLICY_OPTIONS[1];

export interface PolicySelectProps {
  value: string;
  toolName: string;
  saving: boolean;
  onChange: (toolName: string, policy: string) => void;
  size?: "small" | "middle";
  minWidth?: number;
  stopPropagation?: boolean;
}

export const PolicySelect: React.FC<PolicySelectProps> = ({
  value,
  toolName,
  saving,
  onChange,
  size = "small",
  minWidth = 110,
  stopPropagation = true,
}) => {
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
      }}
      styles={{
        selector: {
          backgroundColor: style.bg,
          borderColor: style.border,
          color: style.color,
          borderRadius: 999,
          fontSize: size === "small" ? 11 : 12,
          fontWeight: 600,
          paddingLeft: 8,
          paddingRight: 4,
        },
      }}
      popupMatchSelectWidth={false}
      options={POLICY_OPTIONS.map((o) => ({
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
