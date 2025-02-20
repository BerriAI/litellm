import React, { useState, useEffect } from 'react';
import { Form, Input, InputNumber, Select } from 'antd';
import { TextInput } from "@tremor/react";
import { InfoCircleOutlined } from '@ant-design/icons';
import { Tooltip } from 'antd';

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
  excludedFields?: string[];
  form: any; // Ant Design form instance
}

const SchemaFormFields: React.FC<SchemaFormFieldsProps> = ({ 
  excludedFields = ['key_alias', 'team_id', 'models', 'duration', 'metadata', 'tags', 'guardrails'],
  form 
}) => {
  const [schemaProperties, setSchemaProperties] = useState<OpenAPISchema | null>(null);

  useEffect(() => {
    const fetchOpenAPISchema = async () => {
      try {
        const response = await fetch('http://localhost:4000/openapi.json');
        const schema = await response.json();
        const generateKeyRequest = schema.components.schemas.GenerateKeyRequest;
        setSchemaProperties(generateKeyRequest);
      } catch (error) {
        console.error('Failed to fetch OpenAPI schema:', error);
      }
    };

    fetchOpenAPISchema();
  }, []);

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
    
    // Helper for rendering the label with optional tooltip
    const label = property.description ? (
      <span>
        {property.title || key}{' '}
        <Tooltip title={property.description}>
          <InfoCircleOutlined style={{ marginLeft: '4px' }} />
        </Tooltip>
      </span>
    ) : (property.title || key);

    return (
      <Form.Item
        key={key}
        label={label}
        name={key}
        className="mt-8"
        rules={isRequired ? [{ required: true, message: `${property.title || key} is required` }] : undefined}
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
          <TextInput placeholder={property.description || ''} />
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