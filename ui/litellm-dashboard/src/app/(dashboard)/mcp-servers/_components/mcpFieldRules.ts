import type React from "react";
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
  value: (control.value ?? null) as TValue | null,
  onValueChange: control.onChange,
});

export const selectTriggerControl = (control: MountedFieldControlProps) => ariaOf(control);

const asTagList = (value: unknown): string[] =>
  (Array.isArray(value) ? value : [value]).filter((tag): tag is string => typeof tag === "string" && tag !== "");

// A scope list is delimited on both sides of the field: it is stored as the single space-delimited
// string RFC 6749 defines, and the tag input hands back whatever the admin typed as one custom
// value, so "read,write" would otherwise round-trip as a single malformed scope.
const splitScopes = (value: unknown): string[] => [
  ...new Set(
    asTagList(value)
      .flatMap((entry) => entry.split(/[\s,]+/))
      .filter((scope) => scope !== ""),
  ),
];

const tagField = (control: MountedFieldControlProps, normalise: (value: unknown) => string[]) => {
  const value = normalise(control.value);
  return {
    id: control.id,
    options: value.map((tag) => ({ label: tag, value: tag })),
    value,
    onValueChange: (tags: readonly string[]) => control.onChange(normalise(tags)),
    emptyText: "Type to add",
    allowCustomValues: true,
  };
};

// Tags that are opaque to us: a stdio arg or an access description item may legitimately contain
// spaces and commas, so each entry is stored exactly as the admin typed it.
export const tagsControl = (control: MountedFieldControlProps) => tagField(control, asTagList);

export const scopesControl = (control: MountedFieldControlProps) => tagField(control, splitScopes);

const toNumberOrNull = (raw: string, precision: number | undefined): number | null => {
  if (raw.trim() === "") return null;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return null;
  return precision === undefined ? parsed : Number(parsed.toFixed(precision));
};

export const numberControl = (control: MountedFieldControlProps, precision?: number) => ({
  ...ariaOf(control),
  name: control.name,
  type: "number" as const,
  value: control.value === null || control.value === undefined ? "" : String(control.value),
  onChange: (event: React.ChangeEvent<HTMLInputElement>) =>
    control.onChange(toNumberOrNull(event.target.value, precision)),
});

export const switchControl = (control: MountedFieldControlProps) => ({
  ...ariaOf(control),
  checked: control.value === true,
  onCheckedChange: (checked: boolean) => control.onChange(checked),
});

export const invertedSwitchControl = (control: MountedFieldControlProps) => ({
  ...ariaOf(control),
  checked: control.value !== true,
  onCheckedChange: (checked: boolean) => control.onChange(!checked),
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
