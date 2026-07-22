"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm, type FieldValues, type UseFormProps, type UseFormReturn } from "react-hook-form";
import type { $ZodType } from "zod/v4/core";

export const useZodForm = <TInput extends FieldValues, TOutput extends FieldValues>(
  schema: $ZodType<TOutput, TInput>,
  props?: Omit<UseFormProps<TInput, unknown, TOutput>, "resolver">,
): UseFormReturn<TInput, unknown, TOutput> =>
  useForm<TInput, unknown, TOutput>({
    ...props,
    resolver: zodResolver(schema),
  });
