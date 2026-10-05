"use client";

import { X } from "lucide-react";
import { Controller, useFieldArray, useFormContext } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { FieldError } from "@/components/ui/field";
import { Textarea } from "@/components/ui/textarea";

import { WatchPicker } from "../WatchPicker";
import type { InvestigationInput } from "../investigationSchema";

export function ExpectationsFields() {
  const {
    control,
    register,
    formState: { errors },
  } = useFormContext<InvestigationInput>();
  const { fields, append, remove } = useFieldArray({ control, name: "questions", keyName: "fieldId" });
  return (
    <>
      <label className="grid gap-2">
        <span className="text-sm font-medium">What should the agent be doing?</span>
        <Textarea
          {...register("context")}
          rows={3}
          placeholder="Answer the customer's question using verified sources and explain when information is missing."
        />
        <FieldError>{errors.context?.message}</FieldError>
      </label>
      <Controller
        control={control}
        name="watching"
        render={({ field }) => (
          <WatchPicker
            selected={new Set(field.value)}
            onChange={(values) => field.onChange([...values])}
            onAddCustom={() => append({ id: crypto.randomUUID(), instruction: "", enabled: true })}
          />
        )}
      />
      {fields.length > 0 && (
        <fieldset className="grid gap-2">
          <legend className="sr-only">Custom checks</legend>
          {fields.map((check, index) => (
            <div key={check.fieldId} className="flex items-start gap-1.5">
              <div className="grid min-w-0 flex-1 gap-1">
                <Textarea
                  aria-label={`Check ${index + 1}`}
                  {...register(`questions.${index}.instruction`)}
                  rows={2}
                  placeholder="e.g. Quotes a price without checking the pricing tool"
                />
                <FieldError>{errors.questions?.[index]?.instruction?.message}</FieldError>
              </div>
              <Button
                variant="ghost"
                size="icon-sm"
                className="mt-1 text-muted-foreground"
                aria-label={`Remove check ${index + 1}`}
                onClick={() => remove(index)}
              >
                <X className="size-4" />
              </Button>
            </div>
          ))}
        </fieldset>
      )}
    </>
  );
}
