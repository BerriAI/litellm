import React, { useState, useEffect } from "react";
import { Form, Select, Spin } from "antd";
import { TextInput } from "@tremor/react";
import { GuardrailProviders, guardrail_provider_map } from './guardrail_info_helpers';
import { getGuardrailProviderSpecificParams } from "../networking";

interface GuardrailProviderFieldsProps {
  selectedProvider: string | null;
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
}

interface ProviderParamsResponse {
  [provider: string]: ProviderParam[];
}

const GuardrailProviderFields: React.FC<GuardrailProviderFieldsProps> = ({
  selectedProvider,
  accessToken,
  providerParams: providerParamsProp = null
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
  
  if (!providerFields || providerFields.length === 0) {
    return <div>No configuration fields available for this provider.</div>;
  }

  return (
    <>
      {providerFields.map((field) => (
        <Form.Item
          key={field.param}
          name={field.param}
          label={field.param}
          tooltip={field.description}
          rules={field.required ? [{ required: true, message: `${field.param} is required` }] : undefined}
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
          ) : field.type === "bool" ? (
            <Select
              placeholder={field.description}
              defaultValue={field.default_value}
            >
              <Select.Option value="True">True</Select.Option>
              <Select.Option value="False">False</Select.Option>
            </Select>
          ) : field.param.includes("password") || field.param.includes("secret") ? (
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
      ))}
    </>
  );
};

export default GuardrailProviderFields; 