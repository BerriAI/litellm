"use client";

import { Input } from "@/components/ui/input";

import { DurationInput } from "@/components/shared/DurationInput";

import type { ActivitySelection } from "../../model/types";

export function SampleFields({
  value,
  onChange,
}: {
  value: ActivitySelection;
  onChange: (value: ActivitySelection) => void;
}) {
  return (
    <>
      <DurationInput
        label="Review the last"
        value={value.lookback_hours ?? 24}
        base="hours"
        max={8760}
        onChange={(lookback_hours) => onChange({ ...value, lookback_hours })}
      />
      <label className="grid gap-2 text-sm">
        Sample (%)
        <Input
          type="number"
          min="0.01"
          max="100"
          step="any"
          value={value.sample_percent ?? 100}
          onChange={(e) => onChange({ ...value, sample_percent: Number(e.target.value) })}
        />
      </label>
    </>
  );
}
