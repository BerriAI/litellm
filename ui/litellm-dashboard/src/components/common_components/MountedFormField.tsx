"use client";

import * as React from "react";
import { Controller, type Control, type ControllerProps, type RegisterOptions } from "react-hook-form";

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
  const names = React.useRef<Set<string>>(new Set());
  return React.useMemo(
    () => ({
      register: (name: string) => {
        names.current.add(name);
        return () => {
          names.current.delete(name);
        };
      },
      mountedNames: () => Array.from(names.current),
    }),
    [],
  );
};

export const projectMountedValues = (registry: MountRegistry, store: MountedFormValues): MountedFormValues =>
  Object.fromEntries(registry.mountedNames().map((name) => [name, store[name]]));

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
