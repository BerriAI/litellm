"use client";

import { Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";

import { Input } from "@/components/ui/input";
import { type Sample, type ActivitySelection } from "../../model/types";

import { useFieldArray, useFormContext } from "react-hook-form";
import type { InvestigationInput } from "../investigationSchema";
export function MetadataFilters({
  value,
  onChange,
  attributes,
  keys,
  id,
}: {
  value: ActivitySelection;
  onChange: (value: ActivitySelection) => void;
  attributes: NonNullable<Sample["executions"][number]["metadata"]>;
  keys: string[];
  id: string;
}) {
  const { control } = useFormContext<InvestigationInput>();
  const { fields, append, remove } = useFieldArray({ control, name: "selection.filters", keyName: "fieldId" });
  const filters = value.filters ?? [];
  const edit = (index: number, field: "key" | "value", text: string) =>
    onChange({ ...value, filters: filters.map((f, i) => (i === index ? { ...f, [field]: text } : f)) });
  return (
    <>
      {fields.map((field, index) => (
        <div key={field.fieldId} className="space-y-2">
          <div className="flex gap-2">
            <Input
              aria-label={`Metadata key ${index + 1}`}
              list={`${id}-keys`}
              placeholder="Metadata key"
              value={filters[index].key}
              onChange={(e) => edit(index, "key", e.target.value)}
            />
            <Button
              variant="ghost"
              size="icon"
              aria-label={`Remove condition ${index + 1}`}
              onClick={() => {
                remove(index);
                onChange({ ...value, filters: filters.filter((_, i) => i !== index) });
              }}
            >
              <X className="size-4" />
            </Button>
          </div>
          <Input
            aria-label={`Metadata value ${index + 1}`}
            list={`${id}-values-${index}`}
            placeholder="Equals"
            value={filters[index].value}
            onChange={(e) => edit(index, "value", e.target.value)}
          />
          <datalist id={`${id}-values-${index}`}>
            {[...new Set(attributes.filter((a) => a.key === filters[index].key).map((a) => a.value))]
              .sort()
              .map((v) => (
                <option key={v} value={v} />
              ))}
          </datalist>
        </div>
      ))}
      <datalist id={`${id}-keys`}>
        {keys.map((key) => (
          <option key={key} value={key} />
        ))}
      </datalist>
      <Button
        variant="outline"
        size="sm"
        disabled={filters.length >= 8}
        onClick={() => {
          append({ key: "", value: "" });
          onChange({ ...value, filters: [...filters, { key: "", value: "" }] });
        }}
      >
        <Plus className="size-3" /> Add condition
      </Button>
    </>
  );
}
