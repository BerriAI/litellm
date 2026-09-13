import { CircleMinus, Plus } from "lucide-react";
import React, { useEffect, useRef } from "react";
import {
  useFieldArray,
  type Control,
  type FieldArrayPath,
  type FieldPath,
  type FieldValues,
  type UseFormGetValues,
} from "react-hook-form";
import { z } from "zod/v4";

import { TeamMetadataField } from "@/app/(dashboard)/hooks/teams/useTeamMetadataSchema";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";

export interface MetadataPair {
  key: string;
  value: string;
}

export const metadataPairsSchema = z
  .array(z.object({ key: z.string().min(1, "Missing key"), value: z.string().optional() }))
  .superRefine((pairs, ctx) => {
    pairs.forEach((pair, index) => {
      if (pair.key && pairs.filter((other) => other.key === pair.key).length > 1) {
        ctx.addIssue({ code: "custom", message: "Duplicate key", path: [index, "key"] });
      }
    });
  });

function formatMetadataValue(value: unknown): string {
  if (typeof value !== "string") {
    return JSON.stringify(value) ?? "";
  }
  try {
    JSON.parse(value);
    return JSON.stringify(value);
  } catch {
    return value;
  }
}

function parseMetadataValue(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

export function metadataObjectToPairs(
  metadata: Record<string, unknown> | null | undefined,
  excludedKeys: ReadonlySet<string> = new Set(),
): MetadataPair[] {
  return Object.entries(metadata ?? {})
    .filter(([key]) => !excludedKeys.has(key))
    .map(([key, value]) => ({ key, value: formatMetadataValue(value) }));
}

export function metadataPairsToObject(
  pairs: readonly (Partial<MetadataPair> | undefined)[] | undefined,
): Record<string, unknown> {
  return Object.fromEntries(
    (pairs ?? [])
      .filter((pair): pair is Partial<MetadataPair> & { key: string } => Boolean(pair?.key))
      .map((pair) => [pair.key, parseMetadataValue(pair.value ?? "")]),
  );
}

interface MetadataKeyValueFieldsProps<TFieldValues extends FieldValues> {
  control: Control<TFieldValues>;
  getValues: UseFormGetValues<TFieldValues>;
  name: FieldArrayPath<TFieldValues>;
  schemaFields?: readonly TeamMetadataField[];
  schemaLoading?: boolean;
}

const MetadataKeyValueFields = <TFieldValues extends FieldValues>({
  control,
  getValues,
  name,
  schemaFields = [],
  schemaLoading = false,
}: MetadataKeyValueFieldsProps<TFieldValues>) => {
  const { fields, append, remove } = useFieldArray({ control, name });
  const seededRef = useRef(false);

  useEffect(() => {
    if (seededRef.current || schemaLoading || schemaFields.length === 0) return;
    seededRef.current = true;
    const pairs: (Partial<MetadataPair> | undefined)[] = getValues(name as unknown as FieldPath<TFieldValues>) ?? [];
    if (!Array.isArray(pairs)) return;
    const existingKeys = new Set(pairs.map((pair) => pair?.key).filter(Boolean));
    const seeded = schemaFields
      .filter((field) => !existingKeys.has(field.key))
      .map((field) => ({ key: field.key, value: "" }));
    if (seeded.length > 0) {
      append(seeded as never, { shouldFocus: false });
    }
  }, [append, getValues, name, schemaFields, schemaLoading]);

  if (schemaLoading) {
    return (
      <div data-testid="metadata-schema-skeleton" className="space-y-2">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-2/3" />
      </div>
    );
  }

  return (
    <>
      {fields.map((field, index) => (
        <div key={field.id} className="mb-2 flex items-start gap-2">
          <FormField control={control} name={`${name}.${index}.key` as FieldPath<TFieldValues>}>
            {({ ref, value, ...rest }) => (
              <Input {...rest} ref={ref} value={(value as string) ?? ""} placeholder="Key" />
            )}
          </FormField>
          <FormField control={control} name={`${name}.${index}.value` as FieldPath<TFieldValues>}>
            {({ ref, value, ...rest }) => (
              <Input {...rest} ref={ref} value={(value as string) ?? ""} placeholder="Value" />
            )}
          </FormField>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Remove key-value pair"
            className="mt-1 text-destructive"
            onClick={() => remove(index)}
          >
            <CircleMinus className="size-4" />
          </Button>
        </div>
      ))}
      <Button
        variant="outline"
        className="w-full border-dashed"
        onClick={() => append({ key: "", value: "" } as never, { shouldFocus: false })}
      >
        <Plus className="size-4" />
        Add Key-Value Pair
      </Button>
    </>
  );
};

export default MetadataKeyValueFields;
