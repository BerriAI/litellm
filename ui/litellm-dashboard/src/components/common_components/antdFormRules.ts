import type { Validate } from "react-hook-form";

import type { MountedFormValues } from "./MountedFormField";

interface AntdRuleForm {
  getFieldValue: (name: string) => unknown;
  isFieldTouched?: (name: string) => boolean;
}

interface AntdRule {
  validator: (rule: never, value: never) => Promise<void>;
}

type AntdRuleSource = AntdRule | ((form: AntdRuleForm) => AntdRule);

type MountedValidate = Validate<unknown, MountedFormValues>;

const isBlank = (value: unknown): boolean => value === undefined || value === null || value === "";

const isEmptyList = (value: unknown): boolean => Array.isArray(value) && value.length === 0;

export const antdRequired =
  (message: string): MountedValidate =>
  (value) =>
    isBlank(value) || isEmptyList(value) ? message : true;

const toMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export const antdRules = (...rules: readonly AntdRuleSource[]): Record<string, MountedValidate> =>
  Object.fromEntries(
    rules.map((rule, index) => [
      `antd_${index}`,
      async (value: unknown, values: MountedFormValues) => {
        const resolved = typeof rule === "function" ? rule({ getFieldValue: (name) => values[name] }) : rule;
        const validator = resolved.validator as (rule: unknown, value: unknown) => Promise<void>;
        try {
          await validator(null, value);
          return true;
        } catch (error) {
          return toMessage(error);
        }
      },
    ]),
  );
