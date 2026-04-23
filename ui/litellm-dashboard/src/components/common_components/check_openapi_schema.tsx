import React, { useState, useEffect } from "react";
import { Form, Select } from "antd";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import NumericalInput from "@/components/shared/numerical_input";
import { getOpenAPISchema } from "../networking";
import { formatLabel } from "@/utils/textUtils";

interface SchemaProperty {
  type?: string;
  title?: string;
  description?: string;
  anyOf?: Array<{ type: string }>;
  enum?: string[];
  format?: string;
}

interface OpenAPISchema {
  properties: {
    [key: string]: SchemaProperty;
  };
  required?: string[];
}

interface SchemaFormFieldsProps {
  schemaComponent: string;
  excludedFields?: string[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  form: any;
  overrideLabels?: { [key: string]: string };
  overrideTooltips?: { [key: string]: string };
  customValidation?: {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    [key: string]: (rule: any, value: any) => Promise<void>;
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  defaultValues?: { [key: string]: any };
}

// Define which fields should be parsed as JSON
export const jsonFields = ["metadata", "config", "enforced_params", "aliases"];

const isJSONField = (key: string, property: SchemaProperty): boolean => {
  return jsonFields.includes(key) || property.format === "json";
};

const validateJSON = (value: string): boolean => {
  if (!value) return true;
  try {
    JSON.parse(value);
    return true;
  } catch {
    return false;
  }
};

const getFieldHelp = (
  key: string,
  property: SchemaProperty,
  type: string,
): string => {
  const defaultHelp =
    {
      string: "Text input",
      number: "Numeric input",
      integer: "Whole number input",
      boolean: "True/False value",
    }[type] || "Text input";

  const specificHelp: { [key: string]: string } = {
    max_budget: "Enter maximum budget in USD (e.g., 100.50)",
    budget_duration: "Select a time period for budget reset",
    tpm_limit: "Enter maximum tokens per minute (whole number)",
    rpm_limit: "Enter maximum requests per minute (whole number)",
    duration: "Enter duration (e.g., 30s, 24h, 7d)",
    metadata:
      'Enter JSON object with key-value pairs\nExample: {"team": "research", "project": "nlp"}',
    config: 'Enter configuration as JSON object\nExample: {"setting": "value"}',
    permissions: "Enter comma-separated permission strings",
    enforced_params:
      'Enter parameters as JSON object\nExample: {"param": "value"}',
    blocked: "Enter true/false or specific block conditions",
    aliases:
      'Enter aliases as JSON object\nExample: {"alias1": "value1", "alias2": "value2"}',
    models: "Select one or more model names",
    key_alias: "Enter a unique identifier for this key",
    tags: "Enter comma-separated tag strings",
  };

  const helpText = specificHelp[key] || defaultHelp;

  if (isJSONField(key, property)) {
    return `${helpText}\nMust be valid JSON format`;
  }

  if (property.enum) {
    return `Select from available options\nAllowed values: ${property.enum.join(", ")}`;
  }

  return helpText;
};

const SchemaFormFields: React.FC<SchemaFormFieldsProps> = ({
  schemaComponent,
  excludedFields = [],
  form,
  overrideLabels = {},
  overrideTooltips = {},
  customValidation = {},
  defaultValues = {},
}) => {
  const [schemaProperties, setSchemaProperties] =
    useState<OpenAPISchema | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchOpenAPISchema = async () => {
      try {
        const schema = await getOpenAPISchema();
        const componentSchema = schema.components.schemas[schemaComponent];

        if (!componentSchema) {
          throw new Error(`Schema component "${schemaComponent}" not found`);
        }

        setSchemaProperties(componentSchema);

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const defaultFormValues: { [key: string]: any } = {};
        Object.keys(componentSchema.properties)
          .filter(
            (key) =>
              !excludedFields.includes(key) && defaultValues[key] !== undefined,
          )
          .forEach((key) => {
            defaultFormValues[key] = defaultValues[key];
          });

        form.setFieldsValue(defaultFormValues);
      } catch (error) {
        console.error("Schema fetch error:", error);
        setError(
          error instanceof Error ? error.message : "Failed to fetch schema",
        );
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    };

    fetchOpenAPISchema();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schemaComponent, form, excludedFields]);

  const getPropertyType = (property: SchemaProperty): string => {
    if (property.type) {
      return property.type;
    }
    if (property.anyOf) {
      const types = property.anyOf.map((t) => t.type);
      if (types.includes("number") || types.includes("integer")) return "number";
      if (types.includes("string")) return "string";
    }
    return "string";
  };

  const renderFormItem = (key: string, property: SchemaProperty) => {
    const type = getPropertyType(property);
    const isRequired = schemaProperties?.required?.includes(key);

    const label = overrideLabels[key] || property.title || formatLabel(key);
    const tooltip = overrideTooltips[key] || property.description;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const rules: any[] = [];
    if (isRequired) {
      rules.push({ required: true, message: `${label} is required` });
    }
    if (customValidation[key]) {
      rules.push({ validator: customValidation[key] });
    }
    if (isJSONField(key, property)) {
      rules.push({
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        validator: async (_: any, value: string) => {
          if (value && !validateJSON(value)) {
            throw new Error("Please enter valid JSON");
          }
        },
      });
    }

    const formLabel = tooltip ? (
      <span>
        {label}{" "}
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">{tooltip}</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </span>
    ) : (
      label
    );

    let inputComponent;
    if (isJSONField(key, property)) {
      inputComponent = (
        <Textarea rows={4} placeholder="Enter as JSON" className="font-mono" />
      );
    } else if (property.enum) {
      inputComponent = (
        <Select>
          {property.enum.map((value) => (
            <Select.Option key={value} value={value}>
              {value}
            </Select.Option>
          ))}
        </Select>
      );
    } else if (type === "number" || type === "integer") {
      inputComponent = (
        <NumericalInput
          style={{ width: "100%" }}
          precision={type === "integer" ? 0 : undefined}
        />
      );
    } else if (key === "duration") {
      inputComponent = <Input placeholder="eg: 30s, 30h, 30d" />;
    } else {
      inputComponent = <Input placeholder={tooltip || ""} />;
    }

    return (
      <Form.Item
        key={key}
        label={formLabel}
        name={key}
        className="mt-8"
        rules={rules}
        initialValue={defaultValues[key]}
        help={
          <div className="text-xs text-muted-foreground">
            {getFieldHelp(key, property, type)}
          </div>
        }
      >
        {inputComponent}
      </Form.Item>
    );
  };

  if (error) {
    return <div className="text-destructive">Error: {error}</div>;
  }

  if (!schemaProperties?.properties) {
    return null;
  }

  return (
    <div>
      {Object.entries(schemaProperties.properties)
        .filter(([key]) => !excludedFields.includes(key))
        .map(([key, property]) => renderFormItem(key, property))}
    </div>
  );
};

export default SchemaFormFields;
