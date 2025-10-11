import React, { useState, useEffect } from "react";
import { Form, Select, Spin } from "antd";
import { TextInput } from "@tremor/react";
import {
  guardrail_provider_map,
  populateGuardrailProviders,
  populateGuardrailProviderMap,
} from "./guardrail_info_helpers";
import { getGuardrailProviderSpecificParams } from "../networking";
import NumericalInput from "../shared/numerical_input";

interface GuardrailProviderSpecificFieldsProps {
  selectedProvider: string;
  accessToken?: string | null;
  providerParams?: ProviderParamsResponse | null;
}

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

interface ProviderParamsResponse {
  [provider: string]: { [key: string]: ProviderParam };
}

const GuardrailProviderSpecificFields: React.FC<GuardrailProviderSpecificFieldsProps> = ({
  selectedProvider,
  accessToken,
  providerParams: providerParamsProp = null,
}) => {
  const [loading, setLoading] = useState(false);
  const [providerParams, setProviderParams] = useState<ProviderParamsResponse | null>(providerParamsProp);
  const [error, setError] = useState<string | null>(null);

  // Fetch provider-specific parameters when component mounts
  useEffect(() => {
    if (providerParamsProp) {
      setProviderParams(providerParamsProp);
      return;
    }

    const fetchProviderParams = async () => {
      if (!accessToken) return;

      setLoading(true);
      setError(null);

      try {
        const data = await getGuardrailProviderSpecificParams(accessToken);
        setProviderParams(data);

        // Populate dynamic providers from API response
        populateGuardrailProviders(data);
        populateGuardrailProviderMap(data);
      } catch (error) {
        console.error("Error fetching provider params:", error);
        setError("Failed to load provider parameters");
      } finally {
        setLoading(false);
      }
    };

    if (!providerParamsProp) {
      fetchProviderParams();
    }
  }, [accessToken, providerParamsProp]);

  // If no provider is selected, don't render anything
  if (!selectedProvider) {
    return null;
  }

  // Show loading state
  if (loading) {
    return <Spin tip="Loading provider parameters..." />;
  }

  // Show error state
  if (error) {
    return <div className="text-red-500">{error}</div>;
  }

  // Get the provider key matching the selected provider
  const providerKey = guardrail_provider_map[selectedProvider]?.toLowerCase();

  // Get parameters for the selected provider
  const providerFields = providerParams && providerParams[providerKey];

  if (!providerFields || Object.keys(providerFields).length === 0) {
    return null;
  }

  // Render fields function
  const renderFields = (fields: { [key: string]: ProviderParam }, parentKey = "") => {
    return Object.entries(fields).map(([fieldKey, field]) => {
      const fullFieldKey = parentKey ? `${parentKey}.${fieldKey}` : fieldKey;

      // Handle nested fields
      if (field.type === "nested" && field.fields) {
        return (
          <div key={fullFieldKey}>
            <div className="mb-2 font-medium">{fieldKey}</div>
            <div className="ml-4 border-l-2 border-gray-200 pl-4">{renderFields(field.fields, fullFieldKey)}</div>
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
        >
          {field.type === "select" && field.options ? (
            <Select placeholder={field.description} defaultValue={field.default_value}>
              {field.options.map((option) => (
                <Select.Option key={option} value={option}>
                  {option}
                </Select.Option>
              ))}
            </Select>
          ) : field.type === "multiselect" && field.options ? (
            <Select mode="multiple" placeholder={field.description} defaultValue={field.default_value}>
              {field.options.map((option) => (
                <Select.Option key={option} value={option}>
                  {option}
                </Select.Option>
              ))}
            </Select>
          ) : field.type === "dict" && field.dict_key_options ? (
            <div className="space-y-3">
              <div className="text-sm text-gray-600 mb-3">{field.description}</div>

              {field.dict_key_options.map((key) => (
                <div key={key} className="flex items-center space-x-3">
                  <div className="w-24 font-medium text-sm">{key}</div>
                  <div className="flex-1">
                    <Form.Item name={[fullFieldKey, key]} style={{ marginBottom: 0 }}>
                      {field.dict_value_type === "number" ? (
                        <NumericalInput step={1} width={200} placeholder={`Enter ${key} value`} />
                      ) : field.dict_value_type === "boolean" ? (
                        <Select placeholder={`Select ${key} value`}>
                          <Select.Option value={true}>True</Select.Option>
                          <Select.Option value={false}>False</Select.Option>
                        </Select>
                      ) : (
                        <TextInput placeholder={`Enter ${key} value`} type="text" />
                      )}
                    </Form.Item>
                  </div>
                </div>
              ))}
            </div>
          ) : field.type === "bool" || field.type === "boolean" ? (
            <Select placeholder={field.description} defaultValue={field.default_value}>
              <Select.Option value="true">True</Select.Option>
              <Select.Option value="false">False</Select.Option>
            </Select>
          ) : field.type === "number" ? (
            <NumericalInput step={1} width={400} placeholder={field.description} />
          ) : fieldKey.includes("password") || fieldKey.includes("secret") || fieldKey.includes("key") ? (
            <TextInput placeholder={field.description} type="password" />
          ) : (
            <TextInput placeholder={field.description} type="text" />
          )}
        </Form.Item>
      );
    });
  };

  return <>{renderFields(providerFields)}</>;
};

export default GuardrailProviderSpecificFields;
