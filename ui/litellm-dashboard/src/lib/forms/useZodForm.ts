"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm, type FieldValues, type UseFormProps, type UseFormReturn } from "react-hook-form";
import { z } from "zod";

type FormValuesOf<T> = Extract<T, FieldValues>;

// TInput is what the widgets hold (strings), TOutput is what handleSubmit receives after
// zod runs coerce/transform; typing useForm with z.infer would collapse the two and lie
// about defaultValues. The `unknown` in the middle is RHF's context generic, which we never use.
export function useZodForm<Schema extends z.core.$ZodType>(
  schema: Schema,
  props?: Omit<UseFormProps<FormValuesOf<z.input<Schema>>, unknown, FormValuesOf<z.output<Schema>>>, "resolver">,
): UseFormReturn<FormValuesOf<z.input<Schema>>, unknown, FormValuesOf<z.output<Schema>>>;
export function useZodForm<TInput extends FieldValues, TOutput extends FieldValues>(
  schema: z.core.$ZodType<TOutput, TInput>,
  props?: Omit<UseFormProps<TInput, unknown, TOutput>, "resolver">,
): UseFormReturn<TInput, unknown, TOutput>;
export function useZodForm(
  schema: z.core.$ZodType,
  props?: Omit<UseFormProps<FieldValues, unknown, FieldValues>, "resolver">,
): UseFormReturn<FieldValues, unknown, FieldValues> {
  return useForm<FieldValues, unknown, FieldValues>({
    ...props,
    resolver: zodResolver(schema as z.core.$ZodType<FieldValues, FieldValues>),
  });
}
