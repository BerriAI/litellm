import type { MountedFormValues } from "@/components/common_components/MountedFormField";
import { get, type Validate } from "react-hook-form";
import { z } from "zod";

type FieldValidator = Validate<unknown, MountedFormValues>;
export type FieldValidation = FieldValidator | Record<string, FieldValidator>;

async function firstError(validate: FieldValidation, value: unknown, values: MountedFormValues) {
  for (const rule of typeof validate === "function" ? [validate] : Object.values(validate)) {
    const result = await rule(value, values);
    if (typeof result === "string") return result;
    if (result === false || (Array.isArray(result) && result.every((message) => typeof message === "string"))) {
      return "";
    }
  }
  return undefined;
}

export function deploymentFormSchema(rules: ReadonlyMap<string, FieldValidation>) {
  return z.record(z.string(), z.unknown()).superRefine(async (values, ctx) => {
    const errors = await Promise.all(
      [...rules].map(async ([name, validate]) => ({
        name,
        message: await firstError(validate, get(values, name), values),
      })),
    );
    for (const { name, message } of errors) {
      if (message !== undefined) ctx.addIssue({ code: "custom", path: name.split("."), message });
    }
  });
}
