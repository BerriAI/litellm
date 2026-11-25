import React from "react";
import { type ArrayPath, type Control, type FieldPath, type FieldValues, useFieldArray } from "react-hook-form";
import { CircleMinus, Plus } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Field, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { FormField } from "@/components/shared/form/FormField";
import NumericalInput from "../shared/numerical_input";
import type { BudgetAlertRow } from "./budgetAlertEmails";

export interface BudgetAlertEmailsEditorText {
  readonly thresholdPlaceholder: string;
  readonly recipientsPlaceholder: string;
  readonly removeThreshold: string;
  readonly addThreshold: string;
}

type RowsPath<TFieldValues extends FieldValues> = string extends keyof TFieldValues ? string : ArrayPath<TFieldValues>;

interface BudgetAlertEmailsEditorProps<TFieldValues extends FieldValues> {
  readonly control: Control<TFieldValues>;
  readonly name: RowsPath<TFieldValues>;
  readonly label: React.ReactNode;
  readonly text: BudgetAlertEmailsEditorText;
}

const EMPTY_ROW: BudgetAlertRow = { threshold: null, emails: "" };

export const BudgetAlertEmailsEditor = <TFieldValues extends FieldValues>({
  control,
  name,
  label,
  text,
}: BudgetAlertEmailsEditorProps<TFieldValues>) => {
  const { fields, append, remove } = useFieldArray({ control, name: name as ArrayPath<TFieldValues> });
  const rowField = (index: number, column: keyof BudgetAlertRow) =>
    `${name}.${index}.${column}` as FieldPath<TFieldValues>;

  return (
    <Field>
      <FieldLabel>{label}</FieldLabel>
      {fields.map((row, index) => (
        <div key={row.id} className="mb-2 flex items-start gap-2">
          <FormField control={control} name={rowField(index, "threshold")} className="w-32">
            {({ ref, value, onChange, ...field }) => (
              <NumericalInput
                {...field}
                ref={ref}
                value={value ?? ""}
                onChange={(event: React.ChangeEvent<HTMLInputElement>) =>
                  onChange(event.target.value === "" ? null : Number(event.target.value))
                }
                placeholder={text.thresholdPlaceholder}
                min={1}
                max={100}
                step={1}
              />
            )}
          </FormField>
          <FormField control={control} name={rowField(index, "emails")} className="flex-1">
            {({ ref, value, ...field }) => (
              <Input {...field} ref={ref} value={value ?? ""} placeholder={text.recipientsPlaceholder} />
            )}
          </FormField>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={text.removeThreshold}
            className="mt-1 text-destructive"
            onClick={() => remove(index)}
          >
            <CircleMinus className="size-4" />
          </Button>
        </div>
      ))}
      <Button
        type="button"
        variant="outline"
        className="w-full border-dashed"
        onClick={() => append(EMPTY_ROW as Parameters<typeof append>[0])}
      >
        <Plus className="size-4" />
        {text.addThreshold}
      </Button>
    </Field>
  );
};
