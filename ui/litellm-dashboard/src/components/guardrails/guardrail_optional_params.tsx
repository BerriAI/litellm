import React from 'react';
import { Form, Select, Typography } from 'antd';
import { TextInput } from '@tremor/react';
import NumericalInput from '../shared/numerical_input';

const { Title } = Typography;

interface ProviderParam {
  param: string;
  description: string;
  required: boolean;
  default_value?: string;
  options?: string[];
  type?: string;
  fields?: { [key: string]: ProviderParam };
  dict_key_options?: string[];
  dict_value_type?: string;
}

interface GuardrailOptionalParamsProps {
  optionalParams: ProviderParam;
  parentFieldKey: string;
}

interface DictFieldProps {
  field: ProviderParam;
  fieldKey: string;
  fullFieldKey: string;
}

const DictField: React.FC<DictFieldProps> = ({ field, fieldKey, fullFieldKey }) => {
  const [selectedEntries, setSelectedEntries] = React.useState<Array<{key: string, id: string}>>([]);
  const [availableKeys, setAvailableKeys] = React.useState<string[]>(field.dict_key_options || []);

  const addEntry = (selectedKey: string) => {
    if (!selectedKey) return;
    
    const newEntry = {
      key: selectedKey,
      id: `${selectedKey}_${Date.now()}`
    };
    
    setSelectedEntries([...selectedEntries, newEntry]);
    setAvailableKeys(availableKeys.filter(key => key !== selectedKey));
  };

  const removeEntry = (entryId: string, keyToRemove: string) => {
    setSelectedEntries(selectedEntries.filter(entry => entry.id !== entryId));
    setAvailableKeys([...availableKeys, keyToRemove].sort());
  };

  return (
    <div className="space-y-3">
      <div className="text-sm text-gray-600 mb-3">
        {field.description}
      </div>
      
      {/* Existing entries */}
      {selectedEntries.map((entry) => (
        <div key={entry.id} className="flex items-center space-x-3 p-3 border rounded-lg">
          <div className="w-24 font-medium text-sm">{entry.key}</div>
          <div className="flex-1">
            <Form.Item
              name={[fullFieldKey, entry.key]}
              style={{ marginBottom: 0 }}
            >
              {field.dict_value_type === "number" ? (
                <NumericalInput
                  step={1}
                  width={200}
                  placeholder={`Enter ${entry.key} value`}
                />
              ) : field.dict_value_type === "boolean" ? (
                <Select placeholder={`Select ${entry.key} value`}>
                  <Select.Option value={true}>True</Select.Option>
                  <Select.Option value={false}>False</Select.Option>
                </Select>
              ) : (
                <TextInput
                  placeholder={`Enter ${entry.key} value`}
                  type="text"
                />
              )}
            </Form.Item>
          </div>
          <button 
            type="button"
            className="text-red-500 hover:text-red-700 text-sm"
            onClick={() => removeEntry(entry.id, entry.key)}
          >
            Remove
          </button>
        </div>
      ))}
      
      {/* Add new entry */}
      {availableKeys.length > 0 && (
        <div className="flex items-center space-x-3 mt-2">
          <Select
            placeholder="Select category to configure"
            style={{ width: 200 }}
            onSelect={(value: string | undefined) => value && addEntry(value)}
            value={undefined}
          >
            {availableKeys.map((key) => (
              <Select.Option key={key} value={key}>
                {key}
              </Select.Option>
            ))}
          </Select>
          <span className="text-sm text-gray-500">Select a category to add threshold configuration</span>
        </div>
      )}
    </div>
  );
};

const GuardrailOptionalParams: React.FC<GuardrailOptionalParamsProps> = ({
  optionalParams,
  parentFieldKey
}) => {
  const renderField = (fieldKey: string, field: ProviderParam) => {
    const fullFieldKey = `${parentFieldKey}.${fieldKey}`;
    
    // Handle dict fields separately since they manage their own Form.Items
    if (field.type === "dict" && field.dict_key_options) {
      return (
        <div key={fullFieldKey} className="mb-6">
          <div className="mb-2 font-medium">{fieldKey}</div>
          <DictField
            field={field}
            fieldKey={fieldKey}
            fullFieldKey={fullFieldKey}
          />
        </div>
      );
    }
    
    return (
      <Form.Item
        key={fullFieldKey}
        name={fullFieldKey}
        label={fieldKey}
        tooltip={field.description}
        rules={field.required ? [{ required: true, message: `${fieldKey} is required` }] : undefined}
        className="mb-6"
      >
        {field.type === "select" && field.options ? (
          <Select 
            placeholder={field.description} 
            defaultValue={field.default_value}
          >
            {field.options.map((option) => (
              <Select.Option key={option} value={option}>
                {option}
              </Select.Option>
            ))}
          </Select>
        ) : field.type === "multiselect" && field.options ? (
          <Select 
            mode="multiple"
            placeholder={field.description} 
            defaultValue={field.default_value}
          >
            {field.options.map((option) => (
              <Select.Option key={option} value={option}>
                {option}
              </Select.Option>
            ))}
          </Select>
        ) : field.type === "bool" || field.type === "boolean" ? (
          <Select
            placeholder={field.description}
            defaultValue={field.default_value}
          >
            <Select.Option value="true">True</Select.Option>
            <Select.Option value="false">False</Select.Option>
          </Select>
        ) : field.type === "number" ? (
          <NumericalInput
            step={1}
            width={400}
            placeholder={field.description}
          />
        ) : fieldKey.includes("password") || fieldKey.includes("secret") || fieldKey.includes("key") ? (
          <TextInput
            placeholder={field.description}
            type="password"
          />
        ) : (
          <TextInput
            placeholder={field.description}
            type="text"
          />
        )}
      </Form.Item>
    );
  };

  if (!optionalParams.fields || Object.keys(optionalParams.fields).length === 0) {
    return null;
  }

  return (
    <div className="guardrail-optional-params">
      <div className="mb-6">
        <Title level={4} className="mb-0 font-semibold text-gray-800">
          {optionalParams.description || 'Optional Parameters'}
        </Title>
      </div>
      
      <div className="space-y-4">
        {Object.entries(optionalParams.fields).map(([fieldKey, field]) =>
          renderField(fieldKey, field)
        )}
      </div>
    </div>
  );
};

export default GuardrailOptionalParams; 