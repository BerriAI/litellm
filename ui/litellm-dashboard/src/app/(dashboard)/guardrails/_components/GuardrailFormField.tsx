"use client";

import { CircleHelp } from "lucide-react";
import React, { useId } from "react";
import { useController, type Control, type ControllerRenderProps, type RegisterOptions } from "react-hook-form";
import { Field, FieldDescription, FieldError, FieldLabel } from "@/components/ui/field";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export interface GuardrailCriterion {
  name: string;
  weight: number | string;
  description?: string;
}

export interface GuardrailFormValues extends Record<string, unknown> {
  criteria?: GuardrailCriterion[];
}
export type GuardrailFormControl = Control<GuardrailFormValues>;
export type GuardrailFieldRules = Pick<RegisterOptions<GuardrailFormValues, string>, "validate">;

export type GuardrailFieldControlProps = ControllerRenderProps<GuardrailFormValues, string> & {
  id: string;
  "aria-invalid": true | undefined;
  "aria-describedby": string | undefined;
};

const isBlank = (value: unknown): boolean =>
  value === undefined || value === null || value === "" || (Array.isArray(value) && value.length === 0);

export const requiredRule = (message: string): GuardrailFieldRules => ({
  validate: (value: unknown) => (isBlank(value) ? message : true),
});

export const asText = (value: unknown): string => {
  if (typeof value === "string") return value;
  if (typeof value === "number") return String(value);
  return "";
};

export const asStringArray = (value: unknown): string[] => {
  if (Array.isArray(value)) return value.filter((entry): entry is string => typeof entry === "string");
  if (typeof value === "string" && value !== "") return [value];
  return [];
};

export const readRecord = (source: unknown, key: string): unknown =>
  source !== null && typeof source === "object" ? (source as Record<string, unknown>)[key] : undefined;

export const labelWithHint = (label: React.ReactNode, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent className="max-w-xs">{hint}</TooltipContent>
    </Tooltip>
  </>
);

export interface GuardrailFieldProps {
  control: GuardrailFormControl;
  name: string;
  label?: React.ReactNode;
  description?: React.ReactNode;
  rules?: GuardrailFieldRules;
  defaultValue?: unknown;
  className?: string;
  children: (control: GuardrailFieldControlProps) => React.ReactNode;
}

export const GuardrailField: React.FC<GuardrailFieldProps> = ({
  control,
  name,
  label,
  description,
  rules,
  defaultValue,
  className,
  children,
}) => {
  const reactId = useId();
  const controlId = `${reactId}-control`;
  const descriptionId = `${reactId}-description`;
  const errorId = `${reactId}-error`;
  const { field, fieldState } = useController({ control, name, rules, defaultValue });
  const invalid = fieldState.error !== undefined;
  const describedBy =
    [description !== undefined ? descriptionId : undefined, invalid ? errorId : undefined]
      .filter((id): id is string => id !== undefined)
      .join(" ") || undefined;

  return (
    <Field data-invalid={invalid || undefined} className={className}>
      {label !== undefined && <FieldLabel htmlFor={controlId}>{label}</FieldLabel>}
      {children({ ...field, id: controlId, "aria-invalid": invalid || undefined, "aria-describedby": describedBy })}
      {description !== undefined && <FieldDescription id={descriptionId}>{description}</FieldDescription>}
      <FieldError id={errorId} errors={[fieldState.error]} />
    </Field>
  );
};

const SKIP_MESSAGE_ITEMS = [
  { label: "Use global default", value: "inherit" },
  { label: "Yes — exclude from guardrail scan", value: "yes" },
  { label: "No — always include in scan", value: "no" },
];

export const SkipMessageSelect: React.FC<{ control: GuardrailFieldControlProps }> = ({ control }) => {
  const { id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy } = control;

  return (
    <Select items={SKIP_MESSAGE_ITEMS} value={asText(value) || null} onValueChange={onChange}>
      <SelectTrigger id={id} aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy} className="w-full">
        <SelectValue placeholder="Select an option" />
      </SelectTrigger>
      <SelectContent>
        {SKIP_MESSAGE_ITEMS.map((item) => (
          <SelectItem key={item.value} value={item.value}>
            {item.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};
