"use client";

import { useId } from "react";
import { Controller, useFormContext } from "react-hook-form";
import { DurationInput } from "@/components/shared/DurationInput";
import { FieldError } from "@/components/ui/field";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import type { InvestigationInput } from "../investigationSchema";

const optionalNumber = (value: unknown) => (value == null || value === "" ? null : Number(value));

export function SampleFields() {
  const {
    control,
    register,
    formState: { errors },
  } = useFormContext<InvestigationInput>();
  const selectionErrors = errors.selection;
  const id = useId();
  return (
    <>
      <div className="grid gap-2">
        <Controller
          control={control}
          name="selection.lookback_hours"
          render={({ field }) => (
            <DurationInput label="Review the last" value={field.value ?? 24} base="hours" onChange={field.onChange} />
          )}
        />
        <FieldError>{selectionErrors?.lookback_hours?.message}</FieldError>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="grid content-start gap-2">
          <label htmlFor={`${id}-sample`} className="text-sm font-medium">
            Sample
          </label>
          <InputGroup>
            <InputGroupInput
              id={`${id}-sample`}
              {...register("selection.sample_percent", { valueAsNumber: true })}
              type="number"
              min="0.01"
              max="100"
              step="any"
              className="tabular-nums"
            />
            <InputGroupAddon aria-hidden="true" align="inline-end">
              %
            </InputGroupAddon>
          </InputGroup>
          <FieldError>{selectionErrors?.sample_percent?.message}</FieldError>
        </div>
        <div className="grid content-start gap-2">
          <label htmlFor={`${id}-max`} className="text-sm font-medium">
            At most
          </label>
          <InputGroup>
            <InputGroupInput
              id={`${id}-max`}
              {...register("selection.sample_size", { setValueAs: optionalNumber })}
              type="number"
              min="1"
              placeholder="No limit"
              className="tabular-nums"
            />
            <InputGroupAddon aria-hidden="true" align="inline-end">
              runs
            </InputGroupAddon>
          </InputGroup>
          <FieldError>{selectionErrors?.sample_size?.message}</FieldError>
        </div>
      </div>
      <FieldError>{selectionErrors?.execution_ids?.message}</FieldError>
    </>
  );
}
