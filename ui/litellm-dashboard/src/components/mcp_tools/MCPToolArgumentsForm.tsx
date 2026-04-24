import React, { forwardRef, useEffect, useImperativeHandle, useMemo } from "react";
import { useForm, Controller } from "react-hook-form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";
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

function validateField(
  key: string,
  prop: InputSchemaProperty,
  required: boolean,
  value: any,
): string | true {
  if (required && (value === undefined || value === null || value === "")) {
    return `Please enter ${key}`;
  }
  if (prop.type === "object" || prop.type === "array") {
    if ((value === undefined || value === null || value === "") && !required) {
      return true;
    }
    try {
      const parsed = typeof value === "string" ? JSON.parse(value) : value;
      const isValidObject =
        prop.type === "object" &&
        parsed !== null &&
        typeof parsed === "object" &&
        !Array.isArray(parsed);
      const isValidArray = prop.type === "array" && Array.isArray(parsed);
      if ((prop.type === "object" && isValidObject) || (prop.type === "array" && isValidArray)) {
        return true;
      }
      return prop.type === "object" ? "Please enter a JSON object" : "Please enter a JSON array";
    } catch {
      return "Invalid JSON";
    }
  }
  return true;
}

const MCPToolArgumentsForm = forwardRef<MCPToolArgumentsFormRef, MCPToolArgumentsFormProps>(
  ({ tool, className }, ref) => {
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

    const defaultValues = useMemo(() => {
      const values: Record<string, any> = {};
      if (actualSchema.properties) {
        Object.entries(actualSchema.properties).forEach(([key, prop]) => {
          values[key] = getInitialValueForField(prop);
        });
      }
      return values;
    }, [actualSchema]);

    const form = useForm<Record<string, any>>({
      defaultValues,
      mode: "onSubmit",
    });
    const { control, handleSubmit, reset, trigger, getValues, formState } = form;

    useEffect(() => {
      reset(defaultValues);
    }, [defaultValues, reset, tool]);

    useImperativeHandle(ref, () => ({
      getSubmitValues: async () => {
        const valid = await trigger();
        if (!valid) {
          throw new Error("Validation failed");
        }
        const values = getValues();
        return convertFormValues(values, actualSchema, schema);
      },
    }));

    if (typeof tool.inputSchema === "string") {
      return (
        <form className={className} onSubmit={handleSubmit(() => {})}>
          <div className="space-y-2">
            <Label htmlFor={`${tool.name}-input`}>
              <span className="text-sm font-medium text-foreground">
                Input <span className="text-destructive">*</span>
              </span>
            </Label>
            <Controller
              control={control}
              name="input"
              rules={{ required: "Please enter input for this tool" }}
              render={({ field, fieldState }) => (
                <>
                  <Input
                    id={`${tool.name}-input`}
                    placeholder="Enter input for this tool"
                    {...field}
                    value={field.value ?? ""}
                  />
                  {fieldState.error && (
                    <p className="text-sm text-destructive">{fieldState.error.message}</p>
                  )}
                </>
              )}
            />
          </div>
        </form>
      );
    }

    if (!actualSchema.properties) {
      return (
        <form className={className} onSubmit={handleSubmit(() => {})}>
          <div className="py-4 text-center text-sm text-muted-foreground">
            No parameters required for this tool.
          </div>
        </form>
      );
    }

    return (
      <form className={className} onSubmit={handleSubmit(() => {})}>
        {Object.entries(actualSchema.properties).map(([key, prop]) => {
          const required = !!actualSchema.required?.includes(key);
          const fieldKey = `${tool.name}-${key}`;
          return (
            <div key={fieldKey} className="space-y-2">
              <Label htmlFor={fieldKey}>
                <span className="text-sm font-medium text-foreground flex items-center">
                  {key} {required && <span className="text-destructive">*</span>}
                  {prop.description && (
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="ml-2 h-3.5 w-3.5 text-muted-foreground hover:text-foreground" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs">{prop.description}</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  )}
                </span>
              </Label>
              <Controller
                control={control}
                name={key}
                rules={{
                  validate: (value) => validateField(key, prop, required, value),
                }}
                render={({ field, fieldState }) => {
                  const errorMessage = fieldState.error?.message;
                  if (prop.type === "string" && prop.enum) {
                    return (
                      <>
                        <Select
                          value={field.value ? String(field.value) : ""}
                          onValueChange={(v) => field.onChange(v)}
                        >
                          <SelectTrigger id={fieldKey}>
                            <SelectValue placeholder={`Select ${key}`} />
                          </SelectTrigger>
                          <SelectContent>
                            {prop.enum!.map((v) => (
                              <SelectItem key={String(v)} value={String(v)}>
                                {String(v)}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        {errorMessage && (
                          <p className="text-sm text-destructive">{errorMessage}</p>
                        )}
                      </>
                    );
                  }
                  if (prop.type === "string") {
                    return (
                      <>
                        <Input
                          id={fieldKey}
                          placeholder={prop.description || `Enter ${key}`}
                          {...field}
                          value={field.value ?? ""}
                        />
                        {errorMessage && (
                          <p className="text-sm text-destructive">{errorMessage}</p>
                        )}
                      </>
                    );
                  }
                  if (prop.type === "number" || prop.type === "integer") {
                    return (
                      <>
                        <Input
                          id={fieldKey}
                          type="number"
                          step={prop.type === "integer" ? 1 : "any"}
                          placeholder={prop.description || `Enter ${key}`}
                          value={field.value ?? ""}
                          onChange={(e) => {
                            const v = e.target.value;
                            field.onChange(v === "" ? "" : Number(v));
                          }}
                          onBlur={field.onBlur}
                        />
                        {errorMessage && (
                          <p className="text-sm text-destructive">{errorMessage}</p>
                        )}
                      </>
                    );
                  }
                  if (prop.type === "boolean") {
                    const valueStr =
                      field.value === true ? "true" : field.value === false ? "false" : "";
                    return (
                      <>
                        <Select
                          value={valueStr}
                          onValueChange={(v) => field.onChange(v === "true")}
                        >
                          <SelectTrigger id={fieldKey}>
                            <SelectValue placeholder={`Select ${key}`} />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="true">True</SelectItem>
                            <SelectItem value="false">False</SelectItem>
                          </SelectContent>
                        </Select>
                        {errorMessage && (
                          <p className="text-sm text-destructive">{errorMessage}</p>
                        )}
                      </>
                    );
                  }
                  if (prop.type === "object" || prop.type === "array") {
                    return (
                      <>
                        <Textarea
                          id={fieldKey}
                          rows={prop.type === "object" ? 4 : 3}
                          placeholder={
                            prop.description ||
                            (prop.type === "object"
                              ? `Enter JSON object for ${key}`
                              : `Enter JSON array for ${key}`)
                          }
                          spellCheck={false}
                          className="font-mono"
                          {...field}
                          value={field.value ?? ""}
                        />
                        {errorMessage && (
                          <p className="text-sm text-destructive">{errorMessage}</p>
                        )}
                      </>
                    );
                  }
                  return (
                    <>
                      <Input
                        id={fieldKey}
                        placeholder={prop.description || `Enter ${key}`}
                        {...field}
                        value={field.value ?? ""}
                      />
                      {errorMessage && (
                        <p className="text-sm text-destructive">{errorMessage}</p>
                      )}
                    </>
                  );
                }}
              />
            </div>
          );
        })}
      </form>
    );
  },
);

MCPToolArgumentsForm.displayName = "MCPToolArgumentsForm";

export default MCPToolArgumentsForm;
