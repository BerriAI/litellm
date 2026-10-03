"use client";

import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

import { WatchPicker } from "../WatchPicker";

import { useFieldArray, useFormContext } from "react-hook-form";
import type { InvestigationInput } from "../investigationSchema";

export function ExpectationsStep() {
  const { control, watch, setValue } = useFormContext<InvestigationInput>();
  const { fields, append, remove } = useFieldArray({ control, name: "questions", keyName: "fieldId" });
  const context = watch("context");
  const questions = watch("questions");
  const watching = new Set(watch("watching"));
  const setContext = (value: string) => setValue("context", value);
  const setWatching = (values: ReadonlySet<string>) => setValue("watching", [...values]);
  return (
    <>
      <label className="grid gap-2 text-sm font-medium">
        What should the agent be doing?
        <Textarea
          value={context}
          onChange={(e) => setContext(e.target.value)}
          maxLength={6000}
          rows={4}
          placeholder="Answer the customer's question using verified sources and explain when information is missing."
        />
      </label>
      <WatchPicker
        selected={watching}
        onChange={setWatching}
        onAddCustom={() => append({ id: crypto.randomUUID(), instruction: "", enabled: true })}
      />
      <fieldset className="space-y-2">
        <legend className="sr-only">Custom checks</legend>
        {fields.map((check, index) => (
          <div key={check.fieldId} className="flex items-start gap-2">
            <Textarea
              aria-label={`Check ${index + 1}`}
              value={questions[index].instruction}
              onChange={(event) => setValue(`questions.${index}.instruction`, event.target.value)}
              rows={2}
              placeholder="e.g. Quotes a price without checking the pricing tool"
            />
            <Button variant="ghost" size="icon" aria-label={`Remove check ${index + 1}`} onClick={() => remove(index)}>
              <X className="size-4" />
            </Button>
          </div>
        ))}
      </fieldset>
    </>
  );
}
