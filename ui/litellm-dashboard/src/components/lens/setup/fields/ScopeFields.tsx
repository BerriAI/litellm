"use client";

import { Controller, useFormContext } from "react-hook-form";

import { RunSearch } from "../../traces/list/runSearch/RunSearch";
import type { InvestigationInput } from "../investigationSchema";
import type { ScopeOptions } from "../useMatchingActivity";

/** Which runs to review, written in the same search language as the Traces tab. */
export function ScopeFields({ runs, range }: ScopeOptions) {
  const { control } = useFormContext<InvestigationInput>();
  return (
    <div className="grid gap-2">
      <span className="text-sm font-medium">Runs to review</span>
      <div className="h-9 overflow-hidden rounded-md border border-input shadow-xs focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/50 dark:bg-input/30">
        <Controller
          control={control}
          name="selection.q"
          render={({ field }) => <RunSearch value={field.value} onChange={field.onChange} runs={runs} range={range} />}
        />
      </div>
      <p className="text-xs text-muted-foreground">
        Same search as the Traces tab, like <code className="font-mono">agent:researcher status:error</code>. Leave
        empty to review every run.
      </p>
    </div>
  );
}
