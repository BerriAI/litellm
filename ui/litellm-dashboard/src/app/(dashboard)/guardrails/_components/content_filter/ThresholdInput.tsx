"use client";

import { useState } from "react";

import { Input } from "@/components/ui/input";

interface ThresholdInputProps {
  value: number;
  onValueChange: (value: number | null) => void;
  min: number;
  max: number;
  step: number;
  id?: string;
}

const decimalsOf = (step: number): number => (String(step).split(".")[1] ?? "").length;

const clamp = (value: number, min: number, max: number): number => Math.min(Math.max(value, min), max);

const parseDecimal = (raw: string): number | null => {
  const trimmed = raw.trim();
  if (trimmed === "") return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
};

export const ThresholdInput = ({ value, onValueChange, min, max, step, id }: ThresholdInputProps) => {
  const [draft, setDraft] = useState<string | null>(null);
  const decimals = decimalsOf(step);
  const display = draft ?? value.toFixed(decimals);
  const current = parseDecimal(display);

  const stepBy = (direction: 1 | -1) => {
    const next = clamp(Number(((current ?? value) + direction * step).toFixed(decimals)), min, max);
    setDraft(next.toFixed(decimals));
    onValueChange(next);
  };

  const handleBlur = () => {
    setDraft(null);
    if (current === null) {
      onValueChange(null);
      return;
    }
    const clamped = clamp(current, min, max);
    if (clamped !== current) onValueChange(clamped);
  };

  return (
    <Input
      id={id}
      role="spinbutton"
      inputMode="decimal"
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={current ?? undefined}
      className="w-20"
      value={display}
      onChange={(event) => {
        setDraft(event.target.value);
        onValueChange(parseDecimal(event.target.value));
      }}
      onBlur={handleBlur}
      onKeyDown={(event) => {
        if (event.key === "ArrowUp") {
          event.preventDefault();
          stepBy(1);
        }
        if (event.key === "ArrowDown") {
          event.preventDefault();
          stepBy(-1);
        }
      }}
    />
  );
};
