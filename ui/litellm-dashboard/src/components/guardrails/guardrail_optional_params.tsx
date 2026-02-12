import React from "react";
import { Form, Select, Typography, Input, Button } from "antd";
import NumericalInput from "../shared/numerical_input";

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
  values?: Record<string, any>;
}

interface DictFieldProps {
  field: ProviderParam;
  fieldKey: string;
  fullFieldKey: string | string[];
  value: any | null;
}

const DictField: React.FC<DictFieldProps> = ({ field, fieldKey, fullFieldKey, value }) => {
  const [selectedEntries, setSelectedEntries] = React.useState<Array<{ key: string; id: string }>>([]);
  const [availableKeys, setAvailableKeys] = React.useState<string[]>(field.dict_key_options || []);

  // Initialize selectedEntries and availableKeys based on existing value
  React.useEffect(() => {
    if (value && typeof value === "object") {
      const existingKeys = Object.keys(value);
      const entries = existingKeys.map((key) => ({
        key: key,
        id: `${key}_${Date.now()}_${Math.random()}`,
      }));
      setSelectedEntries(entries);

      const remainingKeys = (field.dict_key_options || []).filter((key) => !existingKeys.includes(key));
      setAvailableKeys(remainingKeys);
    }
  }, [value, field.dict_key_options]);

  const addEntry = (selectedKey: string) => {
    if (!selectedKey) return;

    const newEntry = {
      key: selectedKey,
      id: `${selectedKey}_${Date.now()}`,
    };

    setSelectedEntries([...selectedEntries, newEntry]);
    setAvailableKeys(availableKeys.filter((key) => key !== selectedKey));
  };

  const removeEntry = (entryId: string, keyToRemove: string) => {
    setSelectedEntries(selectedEntries.filter((entry) => entry.id !== entryId));
    setAvailableKeys([...availableKeys, keyToRemove].sort());
  };

  return (
    <div className="space-y-3">
      {/* Existing entries */}
      {selectedEntries.map((entry) => (
        <div key={entry.id} className="flex items-center space-x-3 p-3 border rounded-lg">
          <div className="w-24 font-medium text-sm">{entry.key}</div>
          <div className="flex-1">
            <Form.Item
              name={Array.isArray(fullFieldKey) ? [...fullFieldKey, entry.key] : [fullFieldKey, entry.key]}
              style={{ marginBottom: 0 }}
              initialValue={value && typeof value === "object" ? value[entry.key] : undefined}
              normalize={
                field.dict_value_type === "number"
                  ? (value) => {
                      if (value === null || value === undefined || value === "") return undefined;
                      const num = Number(value);
                      return isNaN(num) ? value : num;
                    }
                  : undefined
              }
            >
              {field.dict_value_type === "number" ? (
                <NumericalInput step={1} width={200} placeholder={`Enter ${entry.key} value`} />
              ) : field.dict_value_type === "boolean" ? (
                <Select placeholder={`Select ${entry.key} value`}>
                  <Select.Option value={true}>True</Select.Option>
                  <Select.Option value={false}>False</Select.Option>
                </Select>
              ) : (
                <Input placeholder={`Enter ${entry.key} value`} />
              )}
            </Form.Item>
          </div>
          <Button
            type="text"
            danger
            size="small"
            onClick={() => removeEntry(entry.id, entry.key)}
          >
            Remove
          </Button>
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
  parentFieldKey,
  values,
}) => {
  const renderField = (fieldKey: string, field: ProviderParam) => {
    const fullFieldKey = `${parentFieldKey}.${fieldKey}`;
    const value = values?.[fieldKey];
    console.log("value", value);
    // Handle dict fields separately since they manage their own Form.Items
    if (field.type === "dict" && field.dict_key_options) {
      return (
        <div key={fullFieldKey} className="mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200">
          <div className="mb-4 font-medium text-gray-900 text-base">{fieldKey}</div>
          <p className="text-sm text-gray-600 mb-4">{field.description}</p>
          <DictField field={field} fieldKey={fieldKey} fullFieldKey={[parentFieldKey, fieldKey]} value={value} />
        </div>
      );
    }

    return (
      <div key={fullFieldKey} className="mb-8 p-6 bg-white rounded-lg border border-gray-200 shadow-sm">
        <Form.Item
          name={[parentFieldKey, fieldKey]}
          label={
            <div className="mb-2">
              <div className="font-medium text-gray-900 text-base">{fieldKey}</div>
              <p className="text-sm text-gray-600 mt-1">{field.description}</p>
            </div>
          }
          rules={field.required ? [{ required: true, message: `${fieldKey} is required` }] : undefined}
          className="mb-0"
          initialValue={value !== undefined ? value : field.default_value}
          normalize={
            field.type === "number"
              ? (value) => {
                  if (value === null || value === undefined || value === "") return undefined;
                  const num = Number(value);
                  return isNaN(num) ? value : num;
                }
              : undefined
          }
        >
          {field.type === "select" && field.options ? (
            <Select placeholder={field.description}>
              {field.options.map((option) => (
                <Select.Option key={option} value={option}>
                  {option}
                </Select.Option>
              ))}
            </Select>
          ) : field.type === "multiselect" && field.options ? (
            <Select mode="multiple" placeholder={field.description}>
              {field.options.map((option) => (
                <Select.Option key={option} value={option}>
                  {option}
                </Select.Option>
              ))}
            </Select>
          ) : field.type === "bool" || field.type === "boolean" ? (
            <Select placeholder={field.description}>
              <Select.Option value="true">True</Select.Option>
              <Select.Option value="false">False</Select.Option>
            </Select>
          ) : field.type === "number" ? (
            <NumericalInput step={1} width={400} placeholder={field.description} />
          ) : fieldKey.includes("password") || fieldKey.includes("secret") || fieldKey.includes("key") ? (
            <Input.Password placeholder={field.description} />
          ) : (
            <Input placeholder={field.description} />
          )}
        </Form.Item>
      </div>
    );
  };

  if (!optionalParams.fields || Object.keys(optionalParams.fields).length === 0) {
    return null;
  }

  return (
    <div className="guardrail-optional-params">
      <div className="mb-8 pb-4 border-b border-gray-100">
        <Title level={3} className="mb-2 font-semibold text-gray-900">
          Optional Parameters
        </Title>
        <p className="text-gray-600 text-sm">
          {optionalParams.description || "Configure additional settings for this guardrail provider"}
        </p>
      </div>

      <div className="space-y-8">
        {Object.entries(optionalParams.fields).map(([fieldKey, field]) => renderField(fieldKey, field))}
      </div>
    </div>
  );
};

export default GuardrailOptionalParams;
