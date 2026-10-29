import type { Validate } from "react-hook-form";

import type { MountedFormValues } from "./MountedFormField";

interface ValidatorRuleForm {
  getFieldValue: (name: string) => unknown;
  isFieldTouched?: (name: string) => boolean;
}

interface ValidatorRule {
  validator: (rule: never, value: never) => Promise<void>;
}

type ValidatorRuleSource = ValidatorRule | ((form: ValidatorRuleForm) => ValidatorRule);

type MountedValidate = Validate<unknown, MountedFormValues>;

const isBlank = (value: unknown): boolean => value === undefined || value === null || value === "";

const isEmptyList = (value: unknown): boolean => Array.isArray(value) && value.length === 0;

export const requiredRule =
  (message: string): MountedValidate =>
  (value) =>
    isBlank(value) || isEmptyList(value) ? message : true;

const toMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export const validatorRules = (...rules: readonly ValidatorRuleSource[]): Record<string, MountedValidate> =>
  Object.fromEntries(
    rules.map((rule, index) => [
      `rule_${index}`,
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
