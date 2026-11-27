import { createContext, useMemo } from "react";
import { get, type Validate } from "react-hook-form";
import { z } from "zod";

import type { MountedFormValues } from "@/components/common_components/MountedFormField";

type FieldValidator = Validate<unknown, MountedFormValues>;
type FieldValidation = FieldValidator | Record<string, FieldValidator>;
type RegisterValidation = (name: string, validate: FieldValidation) => () => void;

export const McpValidationContext = createContext<RegisterValidation | null>(null);

const fieldError = async (validate: FieldValidation, value: unknown, values: MountedFormValues) => {
  for (const check of typeof validate === "function" ? [validate] : Object.values(validate)) {
    const result = await check(value, values);
    if (typeof result === "string") return result;
    if (result === false || (Array.isArray(result) && result.every((entry) => typeof entry === "string"))) return "";
  }
};

export const useMcpFormSchema = () =>
  useMemo(() => {
    const fields = new Map<string, FieldValidation>();
    const registerValidation: RegisterValidation = (name, validate) => {
      fields.set(name, validate);
      return () => {
        if (fields.get(name) === validate) fields.delete(name);
      };
    };
    const schema = z.custom<MountedFormValues>().superRefine(async (values, context) => {
      await Promise.all(
        Array.from(fields, async ([name, validate]) => {
          const message = await fieldError(validate, get(values, name), values);
          if (message !== undefined) context.addIssue({ code: "custom", path: name.split("."), message });
        }),
      );
    });
    return { schema, registerValidation };
  }, []);
