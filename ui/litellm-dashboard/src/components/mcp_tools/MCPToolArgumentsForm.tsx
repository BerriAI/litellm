import React, { forwardRef, useImperativeHandle, useMemo } from "react";
import { Form, Input, InputNumber, Select, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { MCPTool, InputSchema, InputSchemaProperty } from "./types";

const isPlainObject = (value: unknown): value is Record<string, any> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function buildArrayItems(items?: InputSchemaProperty | InputSchemaProperty[]): any[] {
  if (!items) return [];
  if (Array.isArray(items)) {
    return items
      .map((item) => buildDefaultValue(item))
      .filter((value) => value !== undefined);
  }
  const itemDefault = buildDefaultValue(items);
  return itemDefault !== undefined ? [itemDefault] : [];
}

function buildDefaultValue(prop?: InputSchemaProperty, overrideDefault?: any): any {
  if (!prop) return undefined;
  const effectiveDefault = overrideDefault !== undefined ? overrideDefault : prop.default;

  if (prop.type === "object") {
    const base = isPlainObject(effectiveDefault) ? { ...effectiveDefault } : {};
    if (prop.properties) {
      Object.entries(prop.properties).forEach(([childKey, childProp]) => {
        base[childKey] = buildDefaultValue(childProp, base[childKey]);
      });
    }
    return base;
  }

  if (prop.type === "array") {
    if (Array.isArray(effectiveDefault)) {
      const itemSchema = prop.items;
      if (!itemSchema) return effectiveDefault;
      if (effectiveDefault.length === 0) {
        const sample = buildArrayItems(itemSchema);
        return sample.length ? sample : effectiveDefault;
      }
      if (Array.isArray(itemSchema)) {
        return effectiveDefault.map((value, index) => {
          const schema = itemSchema[index] ?? itemSchema[itemSchema.length - 1];
          return buildDefaultValue(schema, value);
        });
      }
      return effectiveDefault.map((value) => buildDefaultValue(itemSchema, value));
    }
    if (effectiveDefault !== undefined) return effectiveDefault;
    return buildArrayItems(prop.items);
  }

  if (effectiveDefault !== undefined) return effectiveDefault;
  switch (prop.type) {
    case "integer":
    case "number":
      return 0;
    case "boolean":
      return false;
    case "string":
    default:
      return "";
  }
}

const getInitialValueForField = (prop: InputSchemaProperty): any => {
  const defaultValue = buildDefaultValue(prop);
  if (prop.type === "object" || prop.type === "array") {
    const fallback = prop.type === "array" ? [] : {};
    return JSON.stringify(defaultValue ?? fallback, null, 2);
  }
  return defaultValue;
};

function convertFormValues(
  values: Record<string, any>,
  actualSchema: InputSchema,
  schema: InputSchema,
): Record<string, any> {
  const convertedValues: Record<string, any> = {};
  const schemaToUse = actualSchema;

  Object.entries(values).forEach(([key, value]) => {
    const prop = schemaToUse.properties?.[key];
    if (prop && value !== null && value !== undefined && value !== "") {
      switch (prop.type) {
        case "boolean":
          convertedValues[key] = value === "true" || value === true;
          break;
        case "number":
        case "integer": {
          const numericValue = Number(value);
          convertedValues[key] = Number.isNaN(numericValue)
            ? value
            : prop.type === "integer"
              ? Math.trunc(numericValue)
              : numericValue;
          break;
        }
        case "object":
        case "array": {
          try {
            const parsed = typeof value === "string" ? JSON.parse(value) : value;
            const isValidObject =
              prop.type === "object" &&
              parsed !== null &&
              typeof parsed === "object" &&
              !Array.isArray(parsed);
            const isValidArray = prop.type === "array" && Array.isArray(parsed);
            if ((prop.type === "object" && isValidObject) || (prop.type === "array" && isValidArray)) {
              convertedValues[key] = parsed;
            } else {
              convertedValues[key] = value;
            }
          } catch {
            convertedValues[key] = value;
          }
          break;
        }
        case "string":
          convertedValues[key] = String(value);
          break;
        default:
          convertedValues[key] = value;
      }
    } else if (value !== null && value !== undefined && value !== "") {
      convertedValues[key] = value;
    }
  });

  const isNestedParams =
    schema.properties?.params?.type === "object" && schema.properties.params.properties;

  return isNestedParams ? { params: convertedValues } : convertedValues;
}

export interface MCPToolArgumentsFormRef {
  getSubmitValues: () => Promise<Record<string, any>>;
}

interface MCPToolArgumentsFormProps {
  tool: MCPTool;
  className?: string;
}

const MCPToolArgumentsForm = forwardRef<MCPToolArgumentsFormRef, MCPToolArgumentsFormProps>(
  ({ tool, className }, ref) => {
    const [form] = Form.useForm();

    const schema: InputSchema = useMemo(() => {
      if (typeof tool.inputSchema === "string") {
        return {
          type: "object",
          properties: {
            input: {
              type: "string",
              description: "Input for this tool",
            },
          },
          required: ["input"],
        };
      }
      return tool.inputSchema as InputSchema;
    }, [tool.inputSchema]);

    const actualSchema: InputSchema = useMemo(() => {
      if (
        schema.properties?.params?.type === "object" &&
        schema.properties.params.properties
      ) {
        return {
          type: "object",
          properties: schema.properties.params.properties,
          required: schema.properties.params.required || [],
        };
      }
      return schema;
    }, [schema]);

    useImperativeHandle(ref, () => ({
      getSubmitValues: async () => {
        const values = await form.validateFields();
        return convertFormValues(values, actualSchema, schema);
      },
    }));

    React.useEffect(() => {
      form.resetFields();
      if (!actualSchema.properties) return;

      const initialValues: Record<string, any> = {};
      Object.entries(actualSchema.properties).forEach(([key, prop]) => {
        initialValues[key] = getInitialValueForField(prop);
      });
      form.setFieldsValue(initialValues);
    }, [form, actualSchema, tool]);

    if (typeof tool.inputSchema === "string") {
      return (
        <Form form={form} layout="vertical" className={className}>
          <Form.Item
            label={
              <span className="text-sm font-medium text-gray-700">
                Input <span className="text-red-500">*</span>
              </span>
            }
            name="input"
            rules={[{ required: true, message: "Please enter input for this tool" }]}
          >
            <Input placeholder="Enter input for this tool" />
          </Form.Item>
        </Form>
      );
    }

    if (!actualSchema.properties) {
      return (
        <Form form={form} layout="vertical" className={className}>
          <div className="py-4 text-center text-sm text-gray-500">
            No parameters required for this tool.
          </div>
        </Form>
      );
    }

    return (
      <Form form={form} layout="vertical" className={className}>
        {Object.entries(actualSchema.properties).map(([key, prop]) => {
          const initialValue = getInitialValueForField(prop);
          const fieldKey = `${tool.name}-${key}`;
          return (
            <Form.Item
              key={fieldKey}
              label={
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  {key} {actualSchema.required?.includes(key) && <span className="text-red-500">*</span>}
                  {prop.description && (
                    <Tooltip title={prop.description}>
                      <InfoCircleOutlined className="ml-2 text-gray-400 hover:text-gray-600" />
                    </Tooltip>
                  )}
                </span>
              }
              name={key}
              initialValue={initialValue}
              rules={[
                {
                  required: actualSchema.required?.includes(key),
                  message: `Please enter ${key}`,
                },
                ...(prop.type === "object" || prop.type === "array"
                  ? [
                      {
                        validator: (_rule: any, value: any) => {
                          if (
                            (value === undefined || value === null || value === "") &&
                            !actualSchema.required?.includes(key)
                          ) {
                            return Promise.resolve();
                          }
                          try {
                            const parsed = typeof value === "string" ? JSON.parse(value) : value;
                            const isValidObject =
                              prop.type === "object" &&
                              parsed !== null &&
                              typeof parsed === "object" &&
                              !Array.isArray(parsed);
                            const isValidArray = prop.type === "array" && Array.isArray(parsed);
                            if (
                              (prop.type === "object" && isValidObject) ||
                              (prop.type === "array" && isValidArray)
                            ) {
                              return Promise.resolve();
                            }
                            return Promise.reject(
                              new Error(
                                prop.type === "object" ? "Please enter a JSON object" : "Please enter a JSON array",
                              ),
                            );
                          } catch {
                            return Promise.reject(new Error("Invalid JSON"));
                          }
                        },
                      },
                    ]
                  : []),
              ]}
            >
              {prop.type === "string" && prop.enum ? (
                <Select
                  placeholder={`Select ${key}`}
                  allowClear={!actualSchema.required?.includes(key)}
                  options={prop.enum.map((v) => ({ value: v, label: v }))}
                />
              ) : prop.type === "string" && !prop.enum ? (
                <Input
                  placeholder={prop.description || `Enter ${key}`}
                  allowClear
                />
              ) : prop.type === "number" || prop.type === "integer" ? (
                <InputNumber
                  step={prop.type === "integer" ? 1 : undefined}
                  placeholder={prop.description || `Enter ${key}`}
                  className="w-full"
                  style={{ width: "100%" }}
                />
              ) : prop.type === "boolean" ? (
                <Select
                  placeholder={`Select ${key}`}
                  allowClear={!actualSchema.required?.includes(key)}
                  options={[
                    { value: true, label: "True" },
                    { value: false, label: "False" },
                  ]}
                />
              ) : (prop.type === "object" || prop.type === "array") ? (
                <Input.TextArea
                  rows={prop.type === "object" ? 4 : 3}
                  placeholder={
                    prop.description ||
                    (prop.type === "object"
                      ? `Enter JSON object for ${key}`
                      : `Enter JSON array for ${key}`)
                  }
                  spellCheck={false}
                  className="font-mono"
                />
              ) : (
                <Input
                  placeholder={prop.description || `Enter ${key}`}
                  allowClear
                />
              )}
            </Form.Item>
          );
        })}
      </Form>
    );
  },
);

MCPToolArgumentsForm.displayName = "MCPToolArgumentsForm";

export default MCPToolArgumentsForm;
