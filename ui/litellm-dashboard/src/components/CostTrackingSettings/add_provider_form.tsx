import React from "react";
import { TextInput, Button } from "@tremor/react";
import { Select as AntdSelect, Form, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import Image from "next/image";
import { Providers, provider_map, providerLogoMap } from "../provider_info_helpers";
import { DiscountConfig } from "./types";
import { handleImageError } from "./provider_display_helpers";

interface AddProviderFormProps {
  discountConfig: DiscountConfig;
  selectedProvider: string | undefined;
  newDiscount: string;
  onProviderChange: (provider: string | undefined) => void;
  onDiscountChange: (discount: string) => void;
  onAddProvider: () => void;
}

const AddProviderForm: React.FC<AddProviderFormProps> = ({
  discountConfig,
  selectedProvider,
  newDiscount,
  onProviderChange,
  onDiscountChange,
  onAddProvider,
}) => {
  return (
    <div className="space-y-6">
      <Form.Item
        label={
          <span className="text-sm font-medium text-gray-700 flex items-center">
            Provider
            <Tooltip title="Select the LLM provider you want to configure a discount for">
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
        }
        rules={[{ required: true, message: "Please select a provider" }]}
      >
        <AntdSelect
          showSearch
          placeholder="Select provider"
          value={selectedProvider}
          onChange={onProviderChange}
          style={{ width: "100%" }}
          size="large"
          optionFilterProp="children"
          filterOption={(input, option) =>
            String(option?.label ?? "").toLowerCase().includes(input.toLowerCase())
          }
        >
          {Object.entries(Providers).map(([providerEnum, providerDisplayName]) => {
            const providerValue = provider_map[providerEnum as keyof typeof provider_map];
            // Only show providers that don't already have a discount configured
            if (providerValue && discountConfig[providerValue]) {
              return null;
            }
            return (
              <AntdSelect.Option key={providerEnum} value={providerEnum} label={providerDisplayName}>
                <div className="flex items-center space-x-2">
                  <Image
                    src={providerLogoMap[providerDisplayName]}
                    alt={`${providerEnum} logo`}
                    width={20}
                    height={20}
                    className="w-5 h-5"
                    onError={(e) => handleImageError(e, providerDisplayName)}
                  />
                  <span>{providerDisplayName}</span>
                </div>
              </AntdSelect.Option>
            );
          })}
        </AntdSelect>
      </Form.Item>

      <Form.Item
        label={
          <span className="text-sm font-medium text-gray-700 flex items-center">
            Discount Percentage
            <Tooltip title="Enter a percentage value (e.g., 5 for 5% discount)">
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
        }
        rules={[{ required: true, message: "Please enter a discount percentage" }]}
      >
        <div className="flex items-center gap-2">
          <TextInput
            placeholder="5"
            value={newDiscount}
            onValueChange={onDiscountChange}
            className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500 flex-1"
          />
          <span className="text-gray-600">%</span>
        </div>
      </Form.Item>

      <div className="flex items-center justify-end space-x-3 pt-6 border-t border-gray-100">
        <Button 
          variant="primary"
          onClick={onAddProvider} 
          disabled={!selectedProvider || !newDiscount}
        >
          Add Provider Discount
        </Button>
      </div>
    </div>
  );
};

export default AddProviderForm;

