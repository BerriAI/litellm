import type { Validate } from "react-hook-form";

import type { MountedFieldControlProps, MountedFormValues } from "@/components/common_components/MountedFormField";

type McpValidate = Validate<unknown, MountedFormValues>;

const ariaOf = (control: MountedFieldControlProps) => ({
  id: control.id,
  onBlur: control.onBlur,
  "aria-required": control["aria-required"],
  "aria-invalid": control["aria-invalid"],
  "aria-describedby": control["aria-describedby"],
});

export const textControl = (control: MountedFieldControlProps) => ({
  ...ariaOf(control),
  name: control.name,
  value: control.value === null || control.value === undefined ? "" : String(control.value),
  onChange: control.onChange,
});

export const selectControl = <TValue = unknown>(control: MountedFieldControlProps) => ({
  ...ariaOf(control),
  value: control.value as TValue,
  onChange: control.onChange,
});

export const numberControl = (control: MountedFieldControlProps) => ({
  ...ariaOf(control),
  value: control.value as number | null | undefined,
  onChange: control.onChange,
});

export const switchControl = (control: MountedFieldControlProps) => ({
  ...ariaOf(control),
  checked: control.value === true,
  onChange: control.onChange,
});

export const invertedSwitchControl = (control: MountedFieldControlProps) => ({
  ...ariaOf(control),
  checked: control.value !== true,
  onChange: (checked: boolean) => control.onChange(!checked),
});

export const valueAt = (values: MountedFormValues, path: readonly string[]): unknown =>
  path.reduce<unknown>(
    (node, segment) => (node === null || node === undefined ? undefined : (node as Record<string, unknown>)[segment]),
    values,
  );

export const parsesAsJson =
  (message: string): McpValidate =>
  (value) => {
    if (typeof value !== "string" || value.trim() === "") {
      return true;
    }
    try {
      JSON.parse(value);
      return true;
    } catch {
      return message;
    }
  };

export const parsesAsJsonObject =
  (message: string, notObjectMessage: string): McpValidate =>
  (value) => {
    if (typeof value !== "string" || value === "") {
      return true;
    }
    try {
      const parsed: unknown = JSON.parse(value);
      return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed) ? true : notObjectMessage;
    } catch {
      return message;
    }
  };

export const matchesPattern =
  (pattern: RegExp, message: string): McpValidate =>
  (value) =>
    typeof value === "string" && value !== "" && !pattern.test(value) ? message : true;

export const notOnlyWhitespace =
  (message: string): McpValidate =>
  (value) =>
    typeof value === "string" && value !== "" && value.trim() === "" ? message : true;

export const requiredWhenSiblingSet =
  (siblingPath: readonly string[], message: string): McpValidate =>
  (value, values) =>
    valueAt(values, siblingPath) && !value ? message : true;

export const requiredUnlessSiblingSet =
  (siblingPath: readonly string[], message: string): McpValidate =>
  (value, values) =>
    value || valueAt(values, siblingPath) ? true : message;
