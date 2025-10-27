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

interface GuardrailProviderFieldsProps {
  selectedProvider: string | null;
  accessToken?: string | null;
  providerParams?: ProviderParamsResponse | null;
  value?: Record<string, any> | null;
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

const GuardrailProviderFields: React.FC<GuardrailProviderFieldsProps> = ({
  selectedProvider,
  accessToken,
  providerParams: providerParamsProp = null,
  value = null,
}) => {
  const [loading, setLoading] = useState(false);
  const [providerParams, setProviderParams] = useState<ProviderParamsResponse | null>(providerParamsProp);
  const [error, setError] = useState<string | null>(null);

  // Fetch provider-specific parameters when component mounts
  useEffect(() => {
    if (providerParamsProp) {
      // Props updated externally
      setProviderParams(providerParamsProp);
      return;
    }

    const fetchProviderParams = async () => {
      if (!accessToken) return;

      setLoading(true);
      setError(null);

      try {
        const data = await getGuardrailProviderSpecificParams(accessToken);
        console.log("Provider params API response:", data);
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

    // Only fetch if not provided via props
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

  // Get the provider key matching the selected provider in the guardrail_provider_map
  const providerKey = guardrail_provider_map[selectedProvider]?.toLowerCase();

  // Get parameters for the selected provider
  const providerFields = providerParams && providerParams[providerKey];

  console.log("Provider key:", providerKey);
  console.log("Provider fields:", providerFields);

  if (!providerFields || Object.keys(providerFields).length === 0) {
    return <div>No configuration fields available for this provider.</div>;
  }

  console.log("Value:", value);
  // Convert object to array of entries and render fields
  const renderFields = (fields: { [key: string]: ProviderParam }, parentKey = "", parentValue?: any) => {
    return Object.entries(fields).map(([fieldKey, field]) => {
      const fullFieldKey = parentKey ? `${parentKey}.${fieldKey}` : fieldKey;
      const fieldValue = parentValue ? parentValue[fieldKey] : value?.[fieldKey];
      console.log("Field value:", fieldValue);
      // Skip ui_friendly_name - it's metadata for the UI dropdown, not a user configuration field
      if (fieldKey === "ui_friendly_name") {
        return null;
      }

      // Skip optional_params - they are handled in a separate step
      if (fieldKey === "optional_params" && field.type === "nested" && field.fields) {
        return null;
      }

      // Handle other nested fields (like azure/text_moderations optional_params)
      if (field.type === "nested" && field.fields) {
        return (
          <div key={fullFieldKey}>
            <div className="mb-2 font-medium">{fieldKey}</div>
            <div className="ml-4 border-l-2 border-gray-200 pl-4">
              {renderFields(field.fields, fullFieldKey, fieldValue)}
            </div>
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
            <Select placeholder={field.description} defaultValue={fieldValue || field.default_value}>
              {field.options.map((option) => (
                <Select.Option key={option} value={option}>
                  {option}
                </Select.Option>
              ))}
            </Select>
          ) : field.type === "multiselect" && field.options ? (
            <Select mode="multiple" placeholder={field.description} defaultValue={fieldValue || field.default_value}>
              {field.options.map((option) => (
                <Select.Option key={option} value={option}>
                  {option}
                </Select.Option>
              ))}
            </Select>
          ) : field.type === "bool" || field.type === "boolean" ? (
            <Select
              placeholder={field.description}
              defaultValue={fieldValue !== undefined ? String(fieldValue) : field.default_value}
            >
              <Select.Option value="true">True</Select.Option>
              <Select.Option value="false">False</Select.Option>
            </Select>
          ) : field.type === "number" ? (
            <NumericalInput
              step={1}
              width={400}
              placeholder={field.description}
              defaultValue={fieldValue !== undefined ? Number(fieldValue) : undefined}
            />
          ) : fieldKey.includes("password") || fieldKey.includes("secret") || fieldKey.includes("key") ? (
            <TextInput placeholder={field.description} type="password" defaultValue={fieldValue || ""} />
          ) : (
            <TextInput placeholder={field.description} type="text" defaultValue={fieldValue || ""} />
          )}
        </Form.Item>
      );
    });
  };

  return <>{renderFields(providerFields)}</>;
};

export default GuardrailProviderFields;
