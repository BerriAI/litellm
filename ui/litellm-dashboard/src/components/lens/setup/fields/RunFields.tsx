"use client";

import { Controller, useFormContext, useWatch } from "react-hook-form";
import { Input } from "@/components/ui/input";
import { DurationInput } from "@/components/shared/DurationInput";
import type { InvestigationInput } from "../investigationSchema";
import { AnalysisModelField, type AnalysisModelFieldProps } from "./AnalysisModelField";

export function RunFields({ models, gate }: AnalysisModelFieldProps) {
  const {
    control,
    register,
    setValue,
    formState: { errors },
  } = useFormContext<InvestigationInput>();
  const repeat = useWatch({ control, name: "repeat" });
  return (
    <>
      <div className="space-y-3">
        <label className="flex items-center gap-2 text-sm font-medium">
          <input type="checkbox" className="size-4 rounded border-input accent-foreground" {...register("repeat")} />
          Keep watching for new traces
        </label>
        {repeat && (
          <Controller
            control={control}
            name="interval"
            render={({ field }) => (
              <DurationInput label="Check every" value={field.value} onChange={field.onChange} base="minutes" />
            )}
          />
        )}
        {errors.interval?.message && (
          <p role="alert" className="text-sm text-destructive">
            {errors.interval.message}
          </p>
        )}
      </div>
      <details open={!gate.modelValid || undefined}>
        <summary className="cursor-pointer text-sm font-medium">Advanced options</summary>
        <div className="mt-4 space-y-5">
          <AnalysisModelField models={models} gate={gate} />
          <div className="grid gap-5 sm:grid-cols-2">
            <label className="grid content-start gap-2 text-sm font-medium">
              Maximum runs (optional)
              <Input
                {...register("selection.sample_size", {
                  setValueAs: (value: unknown) => (value == null || value === "" ? null : Number(value)),
                })}
                type="number"
                min="1"
                placeholder="No limit"
              />
              {errors.selection?.sample_size?.message && (
                <p role="alert" className="text-sm text-destructive">
                  {errors.selection.sample_size.message}
                </p>
              )}
            </label>
            <label className="flex items-center gap-2 text-sm font-medium">
              <input
                type="checkbox"
                {...register("manualSelection", {
                  onChange: () => setValue("selection.execution_ids", [], { shouldValidate: true }),
                })}
              />
              Choose individual runs
            </label>
          </div>
          {errors.selection?.execution_ids?.message && (
            <p role="alert" className="text-sm text-destructive">
              {errors.selection.execution_ids.message}
            </p>
          )}
          <div className="grid gap-5 sm:grid-cols-2">
            <label className="grid content-start gap-2 text-sm font-medium">
              Monthly limit (USD)
              <Input {...register("budget", { valueAsNumber: true })} type="number" min="0.01" step="1" />
              {errors.budget?.message && (
                <p role="alert" className="text-sm text-destructive">
                  {errors.budget.message}
                </p>
              )}
            </label>
          </div>
        </div>
      </details>
    </>
  );
}
