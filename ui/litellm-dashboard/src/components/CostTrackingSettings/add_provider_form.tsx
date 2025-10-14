import React from "react";
import { Text, TextInput, Button, Grid, Col } from "@tremor/react";
import { Select as AntdSelect } from "antd";
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
    <div className="border-t pt-6 mt-6">
      <div className="mb-3">
        <Text className="font-medium text-gray-900">Add Provider Discount</Text>
        <Text className="text-xs text-gray-500 mt-1">
          Select a provider and set its discount rate
        </Text>
      </div>
      <Grid numItems={3} className="gap-3">
        <Col numColSpan={1}>
          <AntdSelect
            showSearch
            placeholder="Select provider"
            value={selectedProvider}
            onChange={onProviderChange}
            style={{ width: "100%" }}
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
        </Col>
        <Col numColSpan={1}>
          <TextInput
            placeholder="Discount (0.05 for 5%)"
            value={newDiscount}
            onValueChange={onDiscountChange}
          />
        </Col>
        <Col numColSpan={1}>
          <Button 
            onClick={onAddProvider} 
            className="w-full"
            disabled={!selectedProvider || !newDiscount}
          >
            Add Provider
          </Button>
        </Col>
      </Grid>
    </div>
  );
};

export default AddProviderForm;

