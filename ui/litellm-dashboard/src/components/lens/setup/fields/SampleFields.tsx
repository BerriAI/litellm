"use client";

import { useState } from "react";
import { Controller, useFormContext, useWatch } from "react-hook-form";
import { X } from "lucide-react";
import { DurationInput } from "@/components/shared/DurationInput";
import { FieldError } from "@/components/ui/field";
import { Slider } from "@/components/ui/slider";
import type { InvestigationInput } from "../investigationSchema";
import { useDropPicks } from "../useMatchingActivity";

/** Runs the sampling percentage keeps out of `eligible`, before any cap. */
export const sampledRuns = (eligible: number, percent: number): number => Math.ceil((eligible * percent) / 100);

const capFromText = (text: string): number | null => {
  const digits = text.replace(/\D/g, "");
  return digits ? Number(digits) : null;
};

interface RunCountProps {
  /** The cap the user typed, or null to analyze every sampled run. */
  readonly cap: number | null;
  /** Sampled runs before any cap; undefined while the preview loads. */
  readonly sampled: number | undefined;
  /** False while a newer preview count is loading, so `sampled` may be stale. */
  readonly settled: boolean;
  readonly onCapChange: (cap: number | null) => void;
  readonly onBlur: () => void;
}

/** Shows how many runs will be analyzed; typing a smaller number caps it. */
function RunCount({ cap, sampled, settled, onCapChange, onBlur }: RunCountProps) {
  const [typing, setTyping] = useState<string | null>(null);
  const analyzed = sampled === undefined ? cap : Math.min(sampled, cap ?? Infinity);
  const capped = cap != null && (sampled === undefined || cap < sampled);
  const shown = typing ?? analyzed?.toLocaleString() ?? "";
  const finishTyping = () => {
    setTyping(null);
    const limitsNothing = cap != null && sampled !== undefined && cap >= sampled;
    if (settled && limitsNothing) onCapChange(null);
    onBlur();
  };
  return (
    <div className="grid justify-items-end gap-1">
      <label className="flex items-center gap-2">
        <input
          inputMode="numeric"
          aria-label="Runs to analyze"
          value={shown}
          placeholder="All"
          onFocus={(event) => event.target.select()}
          onChange={(event) => {
            setTyping(event.target.value);
            onCapChange(capFromText(event.target.value));
          }}
          onBlur={finishTyping}
          style={{ width: `${Math.max((shown || "All").length, 3) + 2}ch` }}
          className="h-8 rounded-md border border-input bg-background px-2 py-0 text-right text-base font-semibold tabular-nums shadow-xs outline-none hover:border-ring/60 focus:border-ring focus:ring-[3px] focus:ring-ring/50 dark:bg-input/30"
        />
        <span className="text-sm text-muted-foreground">runs</span>
      </label>
      {capped && (
        <button
          type="button"
          aria-label="Remove cap"
          onClick={() => onCapChange(null)}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          Capped
          <X aria-hidden="true" className="size-3" />
        </button>
      )}
    </div>
  );
}

export interface SampleFieldsProps {
  /** Runs the search matches; undefined while the preview loads. */
  readonly eligible: number | undefined;
  /** False while a newer preview count is loading. */
  readonly settled: boolean;
}

export function SampleFields({ eligible, settled }: SampleFieldsProps) {
  const {
    control,
    formState: { errors },
  } = useFormContext<InvestigationInput>();
  const dropPicks = useDropPicks();
  const percentValue = useWatch({ control, name: "selection.sample_percent" });
  const selectionErrors = errors.selection;
  const percent = Number.isFinite(percentValue) ? percentValue : 100;
  const sampled = eligible === undefined ? undefined : sampledRuns(eligible, percent);
  return (
    <>
      <div className="grid gap-2">
        <Controller
          control={control}
          name="selection.lookback_hours"
          render={({ field }) => (
            <DurationInput
              label="Review the last"
              value={field.value ?? 24}
              base="hours"
              onChange={(hours) => {
                if (hours !== field.value) dropPicks();
                field.onChange(hours);
              }}
            />
          )}
        />
        <FieldError>{selectionErrors?.lookback_hours?.message}</FieldError>
      </div>
      <div className="grid gap-3 rounded-md border px-3 pt-2.5 pb-3.5">
        <div className="flex items-start justify-between gap-3">
          <div className="grid gap-0.5">
            <span className="text-sm font-medium">Sample</span>
            <span className="text-xs tabular-nums text-muted-foreground">
              {eligible === undefined ? `${percent}% of matching runs` : `${percent}% of ${eligible.toLocaleString()}`}
            </span>
          </div>
          <Controller
            control={control}
            name="selection.sample_size"
            render={({ field }) => (
              <RunCount
                cap={field.value ?? null}
                sampled={sampled}
                settled={settled}
                onCapChange={field.onChange}
                onBlur={field.onBlur}
              />
            )}
          />
        </div>
        <Controller
          control={control}
          name="selection.sample_percent"
          render={({ field }) => (
            <Slider
              thumbLabel="Sample"
              min={1}
              max={100}
              step={1}
              value={[Math.min(100, Math.max(1, Math.round(percent)))]}
              onValueChange={(value) => field.onChange(Array.isArray(value) ? value[0] : value)}
            />
          )}
        />
        <FieldError>{selectionErrors?.sample_percent?.message ?? selectionErrors?.sample_size?.message}</FieldError>
      </div>
      <FieldError>{selectionErrors?.execution_ids?.message}</FieldError>
    </>
  );
}
