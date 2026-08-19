"use client";

import * as React from "react";
import {
  Controller,
  useFieldArray,
  useWatch,
  type Control,
  type ControllerProps,
  type RegisterOptions,
  type UseFormGetValues,
  type UseFormReturn,
} from "react-hook-form";

import { Field, FieldDescription, FieldError, FieldLabel } from "@/components/shared/form/field";

export type MountedFormValues = Record<string, unknown>;

export interface MountRegistry {
  readonly register: (name: string) => () => void;
  readonly mountedNames: () => readonly string[];
  readonly subscribe: (listener: () => void) => () => void;
  readonly version: () => number;
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
    subscribe: missingProvider,
    version: missingProvider,
  },
});

export const MountedFormProvider = MountedFormContext.Provider;

export const useMountedFormContext = (): MountedFormContextValue => React.useContext(MountedFormContext);

export const useMountRegistry = (): MountRegistry => {
  const counts = React.useRef<Map<string, number>>(new Map());
  const listeners = React.useRef<Set<() => void>>(new Set());
  const version = React.useRef(0);
  return React.useMemo(() => {
    const bump = () => {
      version.current += 1;
      listeners.current.forEach((listener) => listener());
    };
    return {
      register: (name: string) => {
        const before = counts.current.get(name) ?? 0;
        counts.current.set(name, before + 1);
        if (before === 0) bump();
        return () => {
          const remaining = (counts.current.get(name) ?? 0) - 1;
          if (remaining > 0) {
            counts.current.set(name, remaining);
            return;
          }
          counts.current.delete(name);
          bump();
        };
      },
      mountedNames: () => Array.from(counts.current.keys()),
      subscribe: (listener: () => void) => {
        listeners.current.add(listener);
        return () => {
          listeners.current.delete(listener);
        };
      },
      version: () => version.current,
    };
  }, []);
};

const isIndexSegment = (segment: string): boolean => /^\d+$/.test(segment);

const readPath = (source: unknown, path: readonly string[]): unknown =>
  path.reduce<unknown>(
    (value, segment) =>
      value === null || value === undefined ? undefined : (value as Record<string, unknown>)[segment],
    source,
  );

const cloneContainer = (target: unknown, head: string): Record<string, unknown> | unknown[] => {
  if (Array.isArray(target)) return [...target];
  if (target !== null && typeof target === "object") return { ...(target as Record<string, unknown>) };
  return isIndexSegment(head) ? [] : {};
};

const writePath = (target: unknown, path: readonly string[], value: unknown): unknown => {
  const [head, ...rest] = path;
  const container = cloneContainer(target, head);
  const next = rest.length === 0 ? value : writePath(readPath(container, [head]), rest, value);
  if (Array.isArray(container)) {
    const copy = [...container];
    copy[Number(head)] = next;
    return copy;
  }
  return { ...container, [head]: next };
};

const collectPaths = (store: unknown, paths: readonly string[], seed: unknown): unknown =>
  paths.reduce<unknown>((acc, path) => {
    const segments = path.split(".");
    return writePath(acc, segments, readPath(store, segments));
  }, seed);

export const projectMountedValues = (
  registry: MountRegistry,
  source: MountedFormValues | UseFormGetValues<MountedFormValues>,
): MountedFormValues =>
  collectPaths(typeof source === "function" ? source() : source, registry.mountedNames(), {}) as MountedFormValues;

export const changedValuesFor = (name: string, store: MountedFormValues): MountedFormValues =>
  collectPaths(store, [name], {}) as MountedFormValues;

const projectSubtree = (mountedNames: readonly string[], name: string, subtree: unknown): unknown => {
  if (mountedNames.includes(name)) {
    return subtree;
  }
  const prefix = `${name}.`;
  const relative = mountedNames
    .filter((mounted) => mounted.startsWith(prefix))
    .map((mounted) => mounted.slice(prefix.length));
  return relative.length === 0 ? undefined : collectPaths(subtree, relative, undefined);
};

const useMountedNames = (registry: MountRegistry): readonly string[] => {
  React.useSyncExternalStore(registry.subscribe, registry.version, registry.version);
  return registry.mountedNames();
};

export const useMountedWatch = (name: string, context?: MountedFormContextValue): unknown => {
  const fallback = React.useContext(MountedFormContext);
  const { control, registry } = context ?? fallback;
  const mountedNames = useMountedNames(registry);
  const subtree = useWatch({ control, name });
  return React.useMemo(() => projectSubtree(mountedNames, name, subtree), [mountedNames, name, subtree]);
};

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

const mergeValues = (target: unknown, patch: unknown): unknown => {
  if (!isPlainObject(target) || !isPlainObject(patch)) {
    return patch;
  }
  return Object.entries(patch).reduce<Record<string, unknown>>(
    (acc, [key, value]) => ({ ...acc, [key]: mergeValues(target[key], value) }),
    { ...target },
  );
};

export const applyFieldValues = (form: UseFormReturn<MountedFormValues>, patch: MountedFormValues): void => {
  const current = form.getValues();
  Object.entries(patch).forEach(([key, value]) => {
    form.setValue(key, mergeValues(current[key], value));
  });
};

export const resetFieldsToDefaults = (
  form: UseFormReturn<MountedFormValues>,
  defaultValues: MountedFormValues,
  names: readonly string[],
): void => {
  names.forEach((name) => {
    form.setValue(name, readPath(defaultValues, name.split(".")));
    form.clearErrors(name);
  });
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

export interface MountedFieldArray {
  readonly fields: readonly { readonly id: string }[];
  readonly append: (value: MountedFormValues) => void;
  readonly remove: (index: number) => void;
}

export const useMountedFieldArray = (control: Control<MountedFormValues>, name: string): MountedFieldArray => {
  const { fields, append, remove } = useFieldArray({ control, name: name as never });
  return { fields, append: append as (value: MountedFormValues) => void, remove };
};

export const bindControl = <TValue,>(
  control: MountedFieldControlProps,
): Omit<MountedFieldControlProps, "value"> & {
  value: TValue;
} => ({ ...control, value: control.value as TValue });

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
  const { control, registry } = useMountedFormContext();
  React.useEffect(() => registry.register(name), [registry, name]);

  const helpId = `${name}_help`;
  const hasHelp = help !== undefined && help !== null;

  const renderField: ControllerProps<MountedFormValues>["render"] = ({ field, fieldState }) => {
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
