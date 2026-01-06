import React from "react";
import { TextInput, Button } from "@tremor/react";
import { Select as AntdSelect, Form, Tooltip, Radio } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Providers, provider_map, providerLogoMap } from "../provider_info_helpers";
import { MarginConfig } from "./types";
import { handleImageError } from "./provider_display_helpers";

interface AddMarginFormProps {
  marginConfig: MarginConfig;
  selectedProvider: string | undefined;
  marginType: "percentage" | "fixed";
  percentageValue: string;
  fixedAmountValue: string;
  onProviderChange: (provider: string | undefined) => void;
  onMarginTypeChange: (type: "percentage" | "fixed") => void;
  onPercentageChange: (value: string) => void;
  onFixedAmountChange: (value: string) => void;
  onAddProvider: () => void;
}

const AddMarginForm: React.FC<AddMarginFormProps> = ({
  marginConfig,
  selectedProvider,
  marginType,
  percentageValue,
  fixedAmountValue,
  onProviderChange,
  onMarginTypeChange,
  onPercentageChange,
  onFixedAmountChange,
  onAddProvider,
}) => {
  return (
    <div className="space-y-6">
      <Form.Item
        label={
          <span className="text-sm font-medium text-gray-700 flex items-center">
            Provider
            <Tooltip title="Select 'Global' to apply margin to all providers, or select a specific provider">
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
        }
        rules={[{ required: true, message: "Please select a provider" }]}
      >
        <AntdSelect
          showSearch
          placeholder="Select provider or 'Global'"
          value={selectedProvider}
          onChange={onProviderChange}
          style={{ width: "100%" }}
          size="large"
          optionFilterProp="children"
          filterOption={(input, option) =>
            String(option?.label ?? "").toLowerCase().includes(input.toLowerCase())
          }
        >
          <AntdSelect.Option key="global" value="global" label="Global (All Providers)">
            <div className="flex items-center space-x-2">
              <span className="font-medium">Global (All Providers)</span>
            </div>
          </AntdSelect.Option>
          {Object.entries(Providers).map(([providerEnum, providerDisplayName]) => {
            const providerValue = provider_map[providerEnum as keyof typeof provider_map];
            // Only show providers that don't already have a margin configured
            if (providerValue && marginConfig[providerValue]) {
              return null;
            }
            return (
              <AntdSelect.Option key={providerEnum} value={providerEnum} label={providerDisplayName}>
                <div className="flex items-center space-x-2">
                  <img
                    src={providerLogoMap[providerDisplayName]}
                    alt={`${providerEnum} logo`}
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
            Margin Type
            <Tooltip title="Choose how to apply the margin: percentage-based or fixed amount">
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
        }
        rules={[{ required: true, message: "Please select a margin type" }]}
      >
        <Radio.Group
          value={marginType}
          onChange={(e) => onMarginTypeChange(e.target.value)}
          className="w-full"
        >
          <Radio value="percentage">Percentage-based</Radio>
          <Radio value="fixed">Fixed Amount</Radio>
        </Radio.Group>
      </Form.Item>

      {marginType === "percentage" && (
        <Form.Item
          label={
            <span className="text-sm font-medium text-gray-700 flex items-center">
              Margin Percentage
              <Tooltip title="Enter a percentage value (e.g., 10 for 10% margin)">
                <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
              </Tooltip>
            </span>
          }
          rules={[
            { required: true, message: "Please enter a margin percentage" },
            {
              validator: (_, value) => {
                if (!value) {
                  return Promise.reject(new Error("Please enter a margin percentage"));
                }
                const numValue = parseFloat(value);
                if (isNaN(numValue) || numValue < 0 || numValue > 1000) {
                  return Promise.reject(new Error("Percentage must be between 0 and 1000"));
                }
                return Promise.resolve();
              },
            },
          ]}
        >
          <div className="flex items-center gap-2">
            <TextInput
              placeholder="10"
              value={percentageValue}
              onValueChange={onPercentageChange}
              className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500 flex-1"
            />
            <span className="text-gray-600">%</span>
          </div>
        </Form.Item>
      )}

      {marginType === "fixed" && (
        <Form.Item
          label={
            <span className="text-sm font-medium text-gray-700 flex items-center">
              Fixed Margin Amount
              <Tooltip title="Enter a fixed amount in USD (e.g., 0.001 for $0.001 per request)">
                <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
              </Tooltip>
            </span>
          }
          rules={[
            { required: true, message: "Please enter a fixed amount" },
            {
              validator: (_, value) => {
                if (!value) {
                  return Promise.reject(new Error("Please enter a fixed amount"));
                }
                const numValue = parseFloat(value);
                if (isNaN(numValue) || numValue < 0) {
                  return Promise.reject(new Error("Fixed amount must be non-negative"));
                }
                return Promise.resolve();
              },
            },
          ]}
        >
          <div className="flex items-center gap-2">
            <span className="text-gray-600">$</span>
            <TextInput
              placeholder="0.001"
              value={fixedAmountValue}
              onValueChange={onFixedAmountChange}
              className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500 flex-1"
            />
          </div>
        </Form.Item>
      )}

      <div className="flex items-center justify-end space-x-3 pt-6 border-t border-gray-100">
        <Button 
          variant="primary"
          onClick={onAddProvider} 
          disabled={
            !selectedProvider || 
            (marginType === "percentage" && !percentageValue) ||
            (marginType === "fixed" && !fixedAmountValue)
          }
        >
          Add Provider Margin
        </Button>
      </div>
    </div>
  );
};

export default AddMarginForm;

