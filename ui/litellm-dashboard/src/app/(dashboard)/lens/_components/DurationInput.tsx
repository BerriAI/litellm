"use client";

import { useId, useState } from "react";
import { Input } from "@/components/ui/input";

export function DurationInput({
  label,
  value,
  onChange,
  base,
  max,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  base: "minutes" | "hours";
  max: number;
}) {
  const id = useId();
  const units =
    base === "minutes"
      ? [
          { label: "minutes", scale: 1 },
          { label: "hours", scale: 60 },
          { label: "days", scale: 1440 },
        ]
      : [
          { label: "hours", scale: 1 },
          { label: "days", scale: 24 },
        ];
  const [scale, setScale] = useState(() => [...units].reverse().find((unit) => value % unit.scale === 0)?.scale ?? 1);
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="text-sm">
        {label}
      </label>
      <div className="flex gap-2">
        <Input
          id={id}
          type="number"
          min={1 / scale}
          max={max / scale}
          step={1 / scale}
          value={Number.isFinite(value) ? value / scale : ""}
          onChange={(event) => onChange(event.target.value === "" ? NaN : Number(event.target.value) * scale)}
        />
        <select
          aria-label={`${label} unit`}
          value={scale}
          className="h-9 rounded-md border border-input bg-background px-3 text-sm"
          onChange={(event) => {
            const next = Number(event.target.value);
            onChange((value / scale) * next);
            setScale(next);
          }}
        >
          {units.map((unit) => (
            <option key={unit.scale} value={unit.scale}>
              {unit.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
