import { MinusCircleOutlined, PlusOutlined } from "@ant-design/icons";
import { Button, Form, FormInstance, Input, Skeleton, Space } from "antd";
import React, { useEffect, useRef } from "react";

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

interface MetadataKeyValueFieldsProps {
  form: FormInstance;
  name?: string;
  schemaFields?: readonly TeamMetadataField[];
  schemaLoading?: boolean;
}

const MetadataKeyValueFields: React.FC<MetadataKeyValueFieldsProps> = ({
  form,
  name = "metadata",
  schemaFields = [],
  schemaLoading = false,
}) => {
  const seededRef = useRef(false);

  useEffect(() => {
    if (seededRef.current || schemaLoading || schemaFields.length === 0) return;
    seededRef.current = true;
    const pairs: (Partial<MetadataPair> | undefined)[] = form.getFieldValue(name) ?? [];
    if (!Array.isArray(pairs)) return;
    const existingKeys = new Set(pairs.map((pair) => pair?.key).filter(Boolean));
    const seeded = schemaFields
      .filter((field) => !existingKeys.has(field.key))
      .map((field) => ({ key: field.key, value: "" }));
    if (seeded.length > 0) {
      form.setFieldValue(name, [...pairs, ...seeded]);
    }
  }, [form, name, schemaFields, schemaLoading]);

  if (schemaLoading) {
    return (
      <div data-testid="metadata-schema-skeleton">
        <Skeleton active title={false} paragraph={{ rows: 3 }} />
      </div>
    );
  }

  return (
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
  );
};

export default MetadataKeyValueFields;
