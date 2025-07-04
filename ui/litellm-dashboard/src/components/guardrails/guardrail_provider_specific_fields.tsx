import React from 'react';
import { Form, Tooltip } from 'antd';
import { TextInput } from '@tremor/react';
import { GuardrailProviders } from './guardrail_info_helpers';

interface GuardrailProviderSpecificFieldsProps {
  selectedProvider: string;
}

interface FieldConfig {
  key: string;
  label: string;
  placeholder?: string;
  tooltip?: string;
  required?: boolean;
  type?: 'text' | 'password';
}

// Define field configurations for each guardrail provider
const PROVIDER_FIELD_CONFIG: Record<string, FieldConfig[]> = {
  [GuardrailProviders.PresidioPII]: [
    {
      key: 'presidio_analyzer_api_base',
      label: 'Presidio Analyzer API Base',
      placeholder: 'http://analyzer:3000',
      tooltip: 'Base URL for the Presidio Analyzer API',
      required: false,
      type: 'text'
    },
    {
      key: 'presidio_anonymizer_api_base',
      label: 'Presidio Anonymizer API Base',
      placeholder: 'http://anonymizer:3000',
      tooltip: 'Base URL for the Presidio Anonymizer API',
      required: false,
      type: 'text'
    }
  ]
};

const GuardrailProviderSpecificFields: React.FC<GuardrailProviderSpecificFieldsProps> = ({
  selectedProvider
}) => {
  // Get field configurations for the selected provider
  const fieldConfigs = PROVIDER_FIELD_CONFIG[selectedProvider] || [];
  
  if (fieldConfigs.length === 0) {
    return null;
  }

  return (
    <>
      {fieldConfigs.map((field) => (
        <Form.Item 
          key={field.key}
          name={field.key}
          label={field.label}
          tooltip={field.tooltip}
          rules={field.required ? [{ required: true, message: 'Required' }] : undefined}
        >
          <TextInput 
            placeholder={field.placeholder} 
            type={field.type || 'text'} 
          />
        </Form.Item>
      ))}
    </>
  );
};

export default GuardrailProviderSpecificFields; 