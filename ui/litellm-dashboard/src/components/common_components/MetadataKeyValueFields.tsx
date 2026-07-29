import { MinusCircleOutlined, PlusOutlined } from "@ant-design/icons";
import { Button, Form, FormInstance, Input, Skeleton, Space } from "antd";
import React, { useEffect, useMemo } from "react";

import { TeamMetadataField } from "@/app/(dashboard)/hooks/teams/useTeamMetadataSchema";

export interface MetadataPair {
  key: string;
  value: string;
}

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

export function schemaMetadataToObject(
  schemaValues: Record<string, string | undefined> | null | undefined,
): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(schemaValues ?? {})
      .filter((entry): entry is [string, string] => typeof entry[1] === "string" && entry[1].trim() !== "")
      .map(([key, value]) => [key, parseMetadataValue(value)]),
  );
}

interface MetadataKeyValueFieldsProps {
  form: FormInstance;
  name?: string;
  schemaName?: string;
  schemaFields?: readonly TeamMetadataField[];
  schemaLoading?: boolean;
  sourceMetadata?: Record<string, unknown> | null;
}

const MetadataKeyValueFields: React.FC<MetadataKeyValueFieldsProps> = ({
  form,
  name = "metadata",
  schemaName = "schema_metadata",
  schemaFields = [],
  schemaLoading = false,
  sourceMetadata = null,
}) => {
  const schemaKeys = useMemo(() => new Set(schemaFields.map((field) => field.key)), [schemaFields]);

  useEffect(() => {
    if (schemaKeys.size === 0) return;
    const pairs: (Partial<MetadataPair> | undefined)[] = form.getFieldValue(name) ?? [];
    if (!Array.isArray(pairs)) return;
    const remaining = pairs.filter((pair) => !pair?.key || !schemaKeys.has(pair.key));
    if (remaining.length !== pairs.length) {
      form.setFieldValue(name, remaining);
    }
  }, [form, name, schemaKeys]);

  if (schemaLoading) {
    return (
      <div data-testid="metadata-schema-skeleton">
        <Skeleton active title={false} paragraph={{ rows: 3 }} />
      </div>
    );
  }

  return (
    <>
      {schemaFields.map((field) => {
        const label = field.label || field.key;
        const initialRaw =
          sourceMetadata != null && field.key in sourceMetadata ? sourceMetadata[field.key] : undefined;
        return (
          <Space key={field.key} style={{ display: "flex", marginBottom: 8 }} align="baseline">
            <Form.Item>
              <Input value={field.key} disabled aria-label={`${label} key`} />
            </Form.Item>
            <Form.Item
              name={[schemaName, field.key]}
              extra={field.description || undefined}
              initialValue={initialRaw !== undefined ? formatMetadataValue(initialRaw) : undefined}
              rules={field.required ? [{ required: true, message: `${label} is required` }] : undefined}
            >
              <Input placeholder={label} />
            </Form.Item>
            {field.required && (
              <span title="Required" style={{ color: "#ef4444" }}>
                *
              </span>
            )}
          </Space>
        );
      })}
      <Form.List name={name}>
        {(fields, { add, remove }) => (
          <>
            {fields.map(({ key, name: fieldName, ...restField }) => (
              <Space key={key} style={{ display: "flex", marginBottom: 8 }} align="baseline">
                <Form.Item
                  {...restField}
                  name={[fieldName, "key"]}
                  rules={[
                    { required: true, message: "Missing key" },
                    {
                      validator: (_, value) => {
                        if (!value) return Promise.resolve();
                        if (schemaKeys.has(value)) {
                          return Promise.reject(new Error("Key is managed by the fields above"));
                        }
                        const all: (Partial<MetadataPair> | undefined)[] = form.getFieldValue(name) ?? [];
                        const dupes = all.filter((entry) => entry?.key === value);
                        if (dupes.length > 1) {
                          return Promise.reject(new Error("Duplicate key"));
                        }
                        return Promise.resolve();
                      },
                    },
                  ]}
                >
                  <Input placeholder="Key" />
                </Form.Item>
                <Form.Item {...restField} name={[fieldName, "value"]}>
                  <Input placeholder="Value" />
                </Form.Item>
                <MinusCircleOutlined
                  aria-label="Remove key-value pair"
                  onClick={() => remove(fieldName)}
                  style={{ color: "#ef4444" }}
                />
              </Space>
            ))}
            <Form.Item style={{ marginBottom: 0 }}>
              <Button type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                Add Key-Value Pair
              </Button>
            </Form.Item>
          </>
        )}
      </Form.List>
    </>
  );
};

export default MetadataKeyValueFields;
