"use client";

import { useId } from "react";
import { Controller, useFormContext, useWatch } from "react-hook-form";
import { DurationInput } from "@/components/shared/DurationInput";
import { FieldError } from "@/components/ui/field";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import type { InvestigationInput } from "../investigationSchema";
import { SwitchRow } from "./SwitchRow";
import { AnalysisModelField, type AnalysisModelFieldProps } from "./AnalysisModelField";

export function RunFields({ models, gate }: AnalysisModelFieldProps) {
  const {
    control,
    register,
    formState: { errors },
  } = useFormContext<InvestigationInput>();
  const repeat = useWatch({ control, name: "repeat" });
  const id = useId();
  return (
    <>
      <div className="grid gap-4 rounded-md border px-3 py-2.5">
        <Controller
          control={control}
          name="repeat"
          render={({ field }) => (
            <SwitchRow
              label="Keep watching for new traces"
              description={field.value ? "Reviews new runs as they arrive" : "Reviews the runs that match now, once"}
              checked={field.value}
              onCheckedChange={field.onChange}
            />
          )}
        />
        {repeat && (
          <div className="grid gap-2 pb-1">
            <Controller
              control={control}
              name="interval"
              render={({ field }) => (
                <DurationInput label="Check every" value={field.value} onChange={field.onChange} base="minutes" />
              )}
            />
            <FieldError>{errors.interval?.message}</FieldError>
          </div>
        )}
      </div>
      <AnalysisModelField models={models} gate={gate} />
      <div className="grid gap-2">
        <label htmlFor={`${id}-budget`} className="text-sm font-medium">
          Monthly limit
        </label>
        <InputGroup>
          <InputGroupAddon aria-hidden="true">$</InputGroupAddon>
          <InputGroupInput
            id={`${id}-budget`}
            aria-describedby={`${id}-budget-hint`}
            {...register("budget", { valueAsNumber: true })}
            type="number"
            min="0.01"
            step="1"
            className="tabular-nums"
          />
          <InputGroupAddon aria-hidden="true" align="inline-end">
            USD
          </InputGroupAddon>
        </InputGroup>
        <p id={`${id}-budget-hint`} className="text-xs text-muted-foreground">
          Analysis pauses once this month&apos;s spend reaches the limit
        </p>
        <FieldError>{errors.budget?.message}</FieldError>
      </div>
    </>
  );
}
