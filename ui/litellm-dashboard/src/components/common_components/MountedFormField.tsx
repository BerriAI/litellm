"use client";

import * as React from "react";
import {
  Controller,
  type Control,
  type ControllerProps,
  type RegisterOptions,
  type UseFormGetValues,
} from "react-hook-form";

import { Field, FieldDescription, FieldError, FieldLabel } from "@/components/shared/form/field";

export type MountedFormValues = Record<string, unknown>;

export interface MountRegistry {
  readonly register: (name: string) => () => void;
  readonly mountedNames: () => readonly string[];
}

export interface MountedFormContextValue {
  readonly control: Control<MountedFormValues>;
  readonly registry: MountRegistry;
}

const missingProvider = (): never => {
  throw new Error("MountedFormField requires a MountedFormProvider ancestor");
};

const MountedFormContext = React.createContext<MountedFormContextValue>({
  get control(): Control<MountedFormValues> {
    return missingProvider();
  },
  registry: {
    register: missingProvider,
    mountedNames: missingProvider,
  },
});

export const MountedFormProvider = MountedFormContext.Provider;

export const useMountRegistry = (): MountRegistry => {
  const counts = React.useRef<Map<string, number>>(new Map());
  return React.useMemo(
    () => ({
      register: (name: string) => {
        counts.current.set(name, (counts.current.get(name) ?? 0) + 1);
        return () => {
          const remaining = (counts.current.get(name) ?? 0) - 1;
          if (remaining > 0) {
            counts.current.set(name, remaining);
          } else {
            counts.current.delete(name);
          }
        };
      },
      mountedNames: () => Array.from(counts.current.keys()),
    }),
    [],
  );
};

const withIndex = (base: readonly unknown[], index: number, next: unknown): readonly unknown[] =>
  Array.from({ length: Math.max(base.length, index + 1) }, (_, i) => (i === index ? next : base[i]));

const setPath = (target: unknown, segments: readonly string[], value: unknown): unknown => {
  const [head, ...rest] = segments;
  if (/^\d+$/.test(head)) {
    const base: readonly unknown[] = Array.isArray(target) ? target : [];
    const index = Number(head);
    return withIndex(base, index, rest.length === 0 ? value : setPath(base[index], rest, value));
  }
  const base: Record<string, unknown> =
    target !== null && typeof target === "object" && !Array.isArray(target) ? (target as Record<string, unknown>) : {};
  return { ...base, [head]: rest.length === 0 ? value : setPath(base[head], rest, value) };
};

export const projectMountedValues = (
  registry: MountRegistry,
  getValues: UseFormGetValues<MountedFormValues>,
): MountedFormValues => {
  const names = [...registry.mountedNames()];
  const values = getValues(names);
  return names.reduce<MountedFormValues>(
    (projected, name, index) => setPath(projected, name.split("."), values[index]) as MountedFormValues,
    {},
  );
};

export type MountedFieldControlProps = {
  readonly id: string;
  readonly name: string;
  readonly value: unknown;
  readonly onChange: (...event: unknown[]) => void;
  readonly onBlur: () => void;
  readonly "aria-required": "true" | undefined;
  readonly "aria-invalid": "true" | undefined;
  readonly "aria-describedby": string | undefined;
};

export interface MountedFormFieldProps {
  readonly name: string;
  readonly label?: React.ReactNode;
  readonly help?: React.ReactNode;
  readonly required?: boolean;
  readonly rules?: Omit<
    RegisterOptions<MountedFormValues, string>,
    "valueAsNumber" | "valueAsDate" | "setValueAs" | "disabled"
  >;
  readonly defaultValue?: unknown;
  readonly bare?: boolean;
  readonly className?: string;
  readonly children: (control: MountedFieldControlProps) => React.ReactNode;
}

export const MountedFormField: React.FC<MountedFormFieldProps> = ({
  name,
  label,
  help,
  required,
  rules,
  defaultValue,
  bare,
  className,
  children,
}) => {
  const { control, registry } = React.useContext(MountedFormContext);
  React.useEffect(() => registry.register(name), [registry, name]);

  const helpId = `${name}_help`;
  const hasHelp = help !== undefined && help !== null;

  const renderField: ControllerProps<MountedFormValues, string>["render"] = ({ field, fieldState }) => {
    const invalid = fieldState.error !== undefined;
    const controlProps: MountedFieldControlProps = {
      id: name,
      name: field.name,
      value: field.value,
      onChange: field.onChange,
      onBlur: field.onBlur,
      "aria-required": required ? "true" : undefined,
      "aria-invalid": invalid ? "true" : undefined,
      "aria-describedby": hasHelp || invalid ? helpId : undefined,
    };

    if (bare) {
      return <>{children(controlProps)}</>;
    }

    return (
      <Field data-invalid={invalid || undefined} className={className}>
        {label !== undefined && <FieldLabel htmlFor={name}>{label}</FieldLabel>}
        {children(controlProps)}
        {hasHelp ? (
          <FieldDescription id={helpId}>{help}</FieldDescription>
        ) : (
          <FieldError id={helpId} errors={[fieldState.error]} />
        )}
      </Field>
    );
  };

  return <Controller control={control} name={name} rules={rules} defaultValue={defaultValue} render={renderField} />;
};
