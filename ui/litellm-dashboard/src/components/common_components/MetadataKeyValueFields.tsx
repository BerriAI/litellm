import { CircleMinus, Plus } from "lucide-react";
import React, { useEffect, useRef, useState } from "react";
import {
  useFieldArray,
  type Control,
  type FieldArrayPath,
  type FieldPath,
  type FieldValues,
  type UseFormGetValues,
} from "react-hook-form";
import { z } from "zod";

import { TeamMetadataField } from "@/app/(dashboard)/hooks/teams/useTeamMetadataSchema";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
    .filter(([key]) => key !== "" && !excludedKeys.has(key))
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

interface MetadataRowProps<TFieldValues extends FieldValues> {
  control: Control<TFieldValues>;
  name: FieldArrayPath<TFieldValues>;
  index: number;
  rowId: string;
  schemaLabel: string | undefined;
  onRemove: () => void;
}

const MetadataRow = <TFieldValues extends FieldValues>({
  control,
  name,
  index,
  rowId,
  schemaLabel,
  onRemove,
}: MetadataRowProps<TFieldValues>) => {
  return (
    <div className="mb-2 flex items-start gap-2">
      {schemaLabel === undefined ? (
        <FormField control={control} name={`${name}.${index}.key` as FieldPath<TFieldValues>}>
          {({ ref, value, ...rest }) => <Input {...rest} ref={ref} value={(value as string) ?? ""} placeholder="Key" />}
        </FormField>
      ) : (
        <Label
          htmlFor={`${rowId}-value`}
          data-testid="metadata-schema-label"
          className="h-9 w-full items-center truncate px-3 font-medium"
        >
          {schemaLabel}
        </Label>
      )}
      <FormField control={control} name={`${name}.${index}.value` as FieldPath<TFieldValues>}>
        {({ ref, value, id, ...rest }) => (
          <Input
            {...rest}
            id={schemaLabel === undefined ? id : `${rowId}-value`}
            ref={ref}
            value={(value as string) ?? ""}
            placeholder="Value"
          />
        )}
      </FormField>
      {schemaLabel === undefined && (
        <Button
          variant="ghost"
          size="icon"
          aria-label="Remove key-value pair"
          className="mt-1 text-destructive"
          onClick={onRemove}
        >
          <CircleMinus className="size-4" />
        </Button>
      )}
    </div>
  );
};

const MetadataKeyValueFields = <TFieldValues extends FieldValues>({
  control,
  getValues,
  name,
  schemaFields = [],
  schemaLoading = false,
}: MetadataKeyValueFieldsProps<TFieldValues>) => {
  const { fields, append, remove } = useFieldArray({ control, name });
  const seededRef = useRef(false);
  const schemaLabelsByKey = new Map(schemaFields.map((field) => [field.key, field.label || field.key]));
  const schemaReady = !schemaLoading && schemaFields.length > 0;
  const livePairs: readonly (Partial<MetadataPair> | undefined)[] =
    getValues(name as unknown as FieldPath<TFieldValues>) ?? [];
  const [keysAtMount, setKeysAtMount] = useState<ReadonlyMap<string, string | undefined>>(() => new Map());
  const unseenKeys = schemaReady
    ? fields.flatMap((field, index) => (keysAtMount.has(field.id) ? [] : [[field.id, livePairs[index]?.key] as const]))
    : [];
  if (unseenKeys.length > 0) {
    setKeysAtMount(new Map([...keysAtMount, ...unseenKeys]));
  }
  const rowKeysAtMount = fields.map((field, index) =>
    keysAtMount.has(field.id) || !schemaReady ? keysAtMount.get(field.id) : livePairs[index]?.key,
  );
  const schemaLabelAt = (index: number): string | undefined => {
    const key = rowKeysAtMount[index];
    return key === undefined || rowKeysAtMount.indexOf(key) !== index ? undefined : schemaLabelsByKey.get(key);
  };

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
        <MetadataRow
          key={field.id}
          control={control}
          name={name}
          index={index}
          rowId={field.id}
          schemaLabel={schemaLabelAt(index)}
          onRemove={() => remove(index)}
        />
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
