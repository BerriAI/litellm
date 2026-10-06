"use client";

import type { ComponentProps } from "react";
import { useId, useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/cva.config";

export type DurationInputProps = Omit<ComponentProps<"div">, "onChange"> & {
  label: string;
  value: number;
  onChange: (value: number) => void;
  base: "minutes" | "hours";
  max?: number;
};

export function DurationInput({ label, value, onChange, base, max, className, ...props }: DurationInputProps) {
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
    <div {...props} data-slot="duration-input" className={cn("grid gap-2", className)}>
      <label htmlFor={id} className="text-sm font-medium">
        {label}
      </label>
      <div className="flex h-9 min-w-0 rounded-md border border-input shadow-xs transition-[color,box-shadow] has-[input:focus-visible]:border-ring has-[input:focus-visible]:ring-[3px] has-[input:focus-visible]:ring-ring/50 has-[input:invalid]:border-destructive dark:bg-input/30">
        <input
          id={id}
          type="number"
          min={1 / scale}
          max={max === undefined ? undefined : max / scale}
          step={1 / scale}
          value={Number.isFinite(value) ? value / scale : ""}
          onChange={(event) => onChange(event.target.value === "" ? NaN : Number(event.target.value) * scale)}
          className="min-w-0 flex-1 border-0 bg-transparent px-3 py-0 shadow-none focus:ring-0 text-base tabular-nums outline-none md:text-sm"
        />
        <div className="relative shrink-0 border-l border-input">
          <select
            aria-label={`${label} unit`}
            value={scale}
            className="h-full appearance-none rounded-r-md border-0 bg-transparent bg-none py-0 pr-8 pl-3 shadow-none focus:ring-0 text-sm text-muted-foreground outline-none hover:text-foreground focus-visible:text-foreground"
            onChange={(event) => setScale(Number(event.target.value))}
          >
            {units.map((unit) => (
              <option key={unit.scale} value={unit.scale}>
                {unit.label}
              </option>
            ))}
          </select>
          <ChevronDown
            aria-hidden="true"
            className="pointer-events-none absolute top-1/2 right-2.5 size-3.5 -translate-y-1/2 text-muted-foreground"
          />
        </div>
      </div>
    </div>
  );
}
