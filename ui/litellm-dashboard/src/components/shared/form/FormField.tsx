"use client";

import * as React from "react";
import {
  Controller,
  type Control,
  type ControllerRenderProps,
  type FieldPath,
  type FieldValues,
} from "react-hook-form";

import { Field, FieldDescription, FieldError, FieldLabel } from "./field";

export type FormFieldControlProps<
  TFieldValues extends FieldValues,
  TName extends FieldPath<TFieldValues>,
> = ControllerRenderProps<TFieldValues, TName> & {
  id: string;
  "aria-invalid": true | undefined;
  "aria-describedby": string | undefined;
};

export interface FormFieldProps<TFieldValues extends FieldValues, TName extends FieldPath<TFieldValues>> {
  control: Control<TFieldValues>;
  name: TName;
  label?: React.ReactNode;
  description?: React.ReactNode;
  orientation?: "vertical" | "horizontal" | "responsive";
  className?: string;
  children: (control: FormFieldControlProps<TFieldValues, TName>) => React.ReactNode;
}

export const FormField = <TFieldValues extends FieldValues, TName extends FieldPath<TFieldValues>>({
  control,
  name,
  label,
  description,
  orientation,
  className,
  children,
}: FormFieldProps<TFieldValues, TName>) => {
  const reactId = React.useId();
  const controlId = `${reactId}-control`;
  const descriptionId = `${reactId}-description`;
  const errorId = `${reactId}-error`;

  return (
    <Controller
      control={control}
      name={name}
      render={({ field, fieldState }) => {
        const invalid = fieldState.error !== undefined;
        const describedBy =
          [description !== undefined ? descriptionId : undefined, invalid ? errorId : undefined]
            .filter((id): id is string => id !== undefined)
            .join(" ") || undefined;
        const controlProps: FormFieldControlProps<TFieldValues, TName> = {
          ...field,
          id: controlId,
          "aria-invalid": invalid || undefined,
          "aria-describedby": describedBy,
        };

        return (
          <Field orientation={orientation} data-invalid={invalid || undefined} className={className}>
            {label !== undefined && <FieldLabel htmlFor={controlId}>{label}</FieldLabel>}
            {children(controlProps)}
            {description !== undefined && <FieldDescription id={descriptionId}>{description}</FieldDescription>}
            <FieldError id={errorId} errors={[fieldState.error]} />
          </Field>
        );
      }}
    />
  );
};
