import { useWatch } from "react-hook-form";
import type { Control, UseFormReturn } from "react-hook-form";

import {
  projectMountedValues,
  type MountRegistry,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";

export type McpForm = UseFormReturn<MountedFormValues>;

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && Object.getPrototypeOf(value) === Object.prototype;

export const deepMergedFieldsValue = (store: unknown, values: Record<string, unknown>): Record<string, unknown> =>
  Object.entries(values).reduce<Record<string, unknown>>(
    (merged, [key, value]) => ({
      ...merged,
      [key]: isPlainObject(value) ? deepMergedFieldsValue(merged[key], value) : value,
    }),
    isPlainObject(store) ? { ...store } : {},
  );

export const setFieldsValue = (form: McpForm, values: Record<string, unknown>): void => {
  const merged = deepMergedFieldsValue(form.getValues(), values);
  Object.keys(values).forEach((key) => form.setValue(key, merged[key]));
};

export const resetFields = (form: McpForm, names: readonly string[], defaults: MountedFormValues = {}): void => {
  names.forEach((name) => {
    form.setValue(name, defaults[name]);
    form.clearErrors(name);
  });
};

const branchAt = (segments: readonly string[], leaf: unknown): unknown => {
  const [head, ...rest] = segments;
  if (head === undefined) {
    return leaf;
  }
  const child = branchAt(rest, leaf);
  if (!/^\d+$/.test(head)) {
    return { [head]: child };
  }
  const index = Number(head);
  return Array.from({ length: index + 1 }, (_, position) => (position === index ? child : undefined));
};

export const singleBranchChange = (path: string, values: MountedFormValues): Record<string, unknown> => {
  const segments = path.split(".");
  const leaf = segments.reduce<unknown>(
    (node, segment) => (node === null || node === undefined ? undefined : (node as Record<string, unknown>)[segment]),
    values,
  );
  return branchAt(segments, leaf) as Record<string, unknown>;
};

export interface McpStaticHeaderRow {
  header?: string;
  value?: string;
}

export interface McpEnvVarRow {
  name?: string;
  value?: string;
  scope?: string;
  description?: string;
}

export interface McpListValues {
  static_headers: McpStaticHeaderRow[];
  env_vars: McpEnvVarRow[];
}

export interface McpFormSnapshot extends MountedFormValues {
  readonly server_name?: string;
  readonly alias?: string;
  readonly description?: string;
  readonly url?: string;
  readonly spec_path?: string;
  readonly transport?: string;
  readonly auth_type?: string;
  readonly oauth_flow_type?: string;
  readonly credentials?: Record<string, unknown>;
  readonly issuer?: string;
  readonly authorization_url?: string;
  readonly token_url?: string;
  readonly registration_url?: string;
  readonly mcp_access_groups?: string[];
  readonly static_headers?: readonly McpStaticHeaderRow[];
  readonly command?: string;
  readonly args?: string[];
  readonly env?: Record<string, string>;
}

export const allFieldsValue = (form: McpForm): McpFormSnapshot => form.getValues() as McpFormSnapshot;

export const listControl = (control: Control<MountedFormValues>): Control<McpListValues> =>
  control as unknown as Control<McpListValues>;

export const mountedPaths = (registry: MountRegistry): readonly string[] =>
  registry.mountedNames().map((name) => (Array.isArray(name) ? name.join(".") : (name as string)));

export const useMountedValues = (form: McpForm, registry: MountRegistry): MountedFormValues => {
  useWatch({ control: form.control });
  return projectMountedValues(registry, form.getValues);
};
