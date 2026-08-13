"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm, type FieldValues, type UseFormProps, type UseFormReturn } from "react-hook-form";
import type { $ZodType } from "zod/v4/core";

// TInput is what the widgets hold (strings), TOutput is what handleSubmit receives after
// zod runs coerce/transform; typing useForm with z.infer would collapse the two and lie
// about defaultValues. zod's $ZodType declares them as <Output, Input>, in that order.
// The `unknown` in the middle is RHF's context generic, which we never use.
export const useZodForm = <TInput extends FieldValues, TOutput extends FieldValues>(
  schema: $ZodType<TOutput, TInput>,
  props?: Omit<UseFormProps<TInput, unknown, TOutput>, "resolver">,
): UseFormReturn<TInput, unknown, TOutput> =>
  useForm<TInput, unknown, TOutput>({
    ...props,
    resolver: zodResolver(schema),
  });
