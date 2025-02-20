import React, { useState, useEffect } from 'react';
import { Form, Input, InputNumber, Select } from 'antd';
import { TextInput } from "@tremor/react";
import { InfoCircleOutlined } from '@ant-design/icons';
import { Tooltip } from 'antd';
import { getOpenAPISchema } from '../networking';
interface SchemaProperty {
  type?: string;
  title?: string;
  description?: string;
  anyOf?: Array<{ type: string }>;
  enum?: string[];
}

interface OpenAPISchema {
  properties: {
    [key: string]: SchemaProperty;
  };
  required?: string[];
}

interface SchemaFormFieldsProps {
  schemaComponent: string;  // Name of the component in OpenAPI schema (e.g., "GenerateKeyRequest")
  excludedFields?: string[];
  form: any; // Ant Design form instance
  overrideLabels?: { [key: string]: string }; // Optional label overrides
  overrideTooltips?: { [key: string]: string }; // Optional tooltip overrides
  customValidation?: { 
    [key: string]: (rule: any, value: any) => Promise<void> 
  }; // Custom validation rules
  defaultValues?: { [key: string]: any }; // Default values for fields
}

const SchemaFormFields: React.FC<SchemaFormFieldsProps> = ({ 
  schemaComponent,
  excludedFields = [],
  form,
  overrideLabels = {},
  overrideTooltips = {},
  customValidation = {},
  defaultValues = {}
}) => {
  const [schemaProperties, setSchemaProperties] = useState<OpenAPISchema | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchOpenAPISchema = async () => {
      try {
        const schema = await getOpenAPISchema();
        const componentSchema = schema.components.schemas[schemaComponent];
        
        if (!componentSchema) {
          throw new Error(`Schema component "${schemaComponent}" not found`);
        }

        if (!componentSchema) {
          throw new Error(`Schema component "${schemaComponent}" not found`);
        }

        setSchemaProperties(componentSchema);
        
        // Set default values
        const defaultFormValues: { [key: string]: any } = {};
        Object.keys(componentSchema.properties)
          .filter(key => !excludedFields.includes(key) && defaultValues[key] !== undefined)
          .forEach(key => {
            defaultFormValues[key] = defaultValues[key];
          });
        
        form.setFieldsValue(defaultFormValues);
        
      } catch (error) {
        console.error('Schema fetch error:', error);
        setError(error instanceof Error ? error.message : 'Failed to fetch schema');
      }
    };

    fetchOpenAPISchema();
  }, [schemaComponent, form, excludedFields]);

  if (error) {
    return <div className="text-red-500">Error: {error}</div>;
  }

  if (!schemaProperties?.properties) {
    return null;
  }

  const getPropertyType = (property: SchemaProperty): string => {
    if (property.type) {
      return property.type;
    }
    if (property.anyOf) {
      const types = property.anyOf.map(t => t.type);
      if (types.includes('number')) return 'number';
      if (types.includes('string')) return 'string';
    }
    return 'string';
  };

  const renderFormItem = (key: string, property: SchemaProperty) => {
    const type = getPropertyType(property);
    const isRequired = schemaProperties?.required?.includes(key);
    
    // Get custom label and tooltip if provided, otherwise use schema values
    const label = overrideLabels[key] || property.title || key;
    const tooltip = overrideTooltips[key] || property.description;
    
    // Create validation rules
    const rules = [];
    if (isRequired) {
      rules.push({ required: true, message: `${label} is required` });
    }
    if (customValidation[key]) {
      rules.push({ validator: customValidation[key] });
    }

    // Helper for rendering the label with optional tooltip
    const formLabel = tooltip ? (
      <span>
        {label}{' '}
        <Tooltip title={tooltip}>
          <InfoCircleOutlined style={{ marginLeft: '4px' }} />
        </Tooltip>
      </span>
    ) : label;

    return (
      <Form.Item
        key={key}
        label={formLabel}
        name={key}
        className="mt-8"
        rules={rules}
        initialValue={defaultValues[key]}
      >
        {property.enum ? (
          <Select>
            {property.enum.map(value => (
              <Select.Option key={value} value={value}>
                {value}
              </Select.Option>
            ))}
          </Select>
        ) : type === 'number' ? (
          <InputNumber style={{ width: '100%' }} />
        ) : (
          <TextInput placeholder={tooltip || ''} />
        )}
      </Form.Item>
    );
  };

  return (
    <div>
      {Object.entries(schemaProperties.properties)
        .filter(([key]) => !excludedFields.includes(key))
        .map(([key, property]) => renderFormItem(key, property))}
    </div>
  );
};

export default SchemaFormFields;