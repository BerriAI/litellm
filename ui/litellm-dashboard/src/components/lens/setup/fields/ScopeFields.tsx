"use client";

import { Controller, useFormContext } from "react-hook-form";

import { RunSearch } from "../../traces/list/runSearch/RunSearch";
import type { InvestigationInput } from "../investigationSchema";
import { useDropPicks, type ScopeOptions } from "../useMatchingActivity";

/** Which runs to review, written in the same search language as the Traces tab; empty reviews every run. */
export function ScopeFields({ runs, range }: ScopeOptions) {
  const { control } = useFormContext<InvestigationInput>();
  const dropPicks = useDropPicks();
  return (
    <div className="min-h-11 rounded-md border border-input py-1 shadow-xs focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/50 dark:bg-input/30">
      <Controller
        control={control}
        name="selection.q"
        render={({ field }) => (
          <RunSearch
            value={field.value}
            onChange={(q) => {
              if (q !== field.value) dropPicks();
              field.onChange(q);
            }}
            runs={runs}
            range={range}
          />
        )}
      />
    </div>
  );
}
