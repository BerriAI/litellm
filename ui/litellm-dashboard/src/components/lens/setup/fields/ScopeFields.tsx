"use client";

import { Controller, useFormContext } from "react-hook-form";

import { RunSearch } from "../../traces/list/runSearch/RunSearch";
import type { InvestigationInput } from "../investigationSchema";
import type { ScopeOptions } from "../useMatchingActivity";

/** Which runs to review, written in the same search language as the Traces tab. */
export function ScopeFields({ runs, range }: ScopeOptions) {
  const { control } = useFormContext<InvestigationInput>();
  return (
    <div className="grid gap-2 text-sm font-medium">
      Runs to review
      <div className="h-9 rounded-md border">
        <Controller
          control={control}
          name="selection.q"
          render={({ field }) => <RunSearch value={field.value} onChange={field.onChange} runs={runs} range={range} />}
        />
      </div>
      <p className="text-xs font-normal text-muted-foreground">
        Leave empty to review every run, or filter like agent:researcher status:error attr.environment:prod
      </p>
    </div>
  );
}
