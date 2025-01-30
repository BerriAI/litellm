import React from "react";
import { Form } from "antd";
import { Select, SelectItem } from "@tremor/react";
import { Providers, providerLogoMap, provider_map } from "../provider_info_helpers";

interface ProviderSelectionProps {
  selectedProvider: string;
  setSelectedProvider: (provider: string) => void;
  setProviderModelsFn: (provider: string) => void;
}

const ProviderSelection: React.FC<ProviderSelectionProps> = ({
  selectedProvider,
  setSelectedProvider,
  setProviderModelsFn,
}) => {
  const form = Form.useFormInstance();

  return (
    <Form.Item
      rules={[{ required: true, message: "Required" }]}
      label="Provider:"
      name="custom_llm_provider"
      tooltip="E.g. OpenAI, Azure OpenAI, Anthropic, Bedrock, etc."
      labelCol={{ span: 10 }}
      labelAlign="left"
    >
      <Select
        value={selectedProvider as string}
        onChange={(value) => {
          // Set the selected provider
          setSelectedProvider(value as unknown as string);
          // Update provider-specific models
          setProviderModelsFn(provider_map[value as unknown as string]);
          // Reset the 'model' field
          form.setFieldsValue({ model: [] });
          // Reset the 'model_name' field
          form.setFieldsValue({ model_name: undefined });
        }}
      >
        {Object.keys(Providers).map((providerKey) => (
          <SelectItem
            key={providerKey}
            value={Providers[providerKey as keyof typeof Providers]}
            onClick={() => {
              setProviderModelsFn(provider_map[providerKey as keyof typeof Providers]);
              setSelectedProvider(Providers[providerKey as keyof typeof Providers]);
            }}
          >
            <div className="flex items-center space-x-2">
              <img
                src={providerLogoMap[Providers[providerKey as keyof typeof Providers]]}
                alt={`${Providers[providerKey as keyof typeof Providers]} logo`}
                className="w-5 h-5"
                onError={(e) => {
                  // Create a div with provider initial as fallback
                  const target = e.target as HTMLImageElement;
                  const parent = target.parentElement;
                  if (parent) {
                    const fallbackDiv = document.createElement('div');
                    fallbackDiv.className = 'w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center text-xs';
                    fallbackDiv.textContent = Providers[providerKey as keyof typeof Providers].charAt(0);
                    parent.replaceChild(fallbackDiv, target);
                  }
                }}
              />
              <span>{Providers[providerKey as keyof typeof Providers]}</span>
            </div>
          </SelectItem>
        ))}
      </Select>
    </Form.Item>
  );
};

export default ProviderSelection;