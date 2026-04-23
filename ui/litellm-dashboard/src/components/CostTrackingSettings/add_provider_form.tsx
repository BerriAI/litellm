import React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Select as AntdSelect, Form } from "antd";
import { Info } from "lucide-react";
import {
  Providers,
  provider_map,
  providerLogoMap,
} from "../provider_info_helpers";
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
          <span className="text-sm font-medium text-foreground flex items-center">
            Provider
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="ml-2 h-3 w-3 text-primary cursor-help" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  Select the LLM provider you want to configure a discount for
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
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
            String(option?.label ?? "")
              .toLowerCase()
              .includes(input.toLowerCase())
          }
        >
          {Object.entries(Providers).map(
            ([providerEnum, providerDisplayName]) => {
              const providerValue =
                provider_map[providerEnum as keyof typeof provider_map];
              if (providerValue && discountConfig[providerValue]) {
                return null;
              }
              return (
                <AntdSelect.Option
                  key={providerEnum}
                  value={providerEnum}
                  label={providerDisplayName}
                >
                  <div className="flex items-center space-x-2">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
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
            },
          )}
        </AntdSelect>
      </Form.Item>

      <Form.Item
        label={
          <span className="text-sm font-medium text-foreground flex items-center">
            Discount Percentage
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="ml-2 h-3 w-3 text-primary cursor-help" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  Enter a percentage value (e.g., 5 for 5% discount)
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </span>
        }
        rules={[{ required: true, message: "Please enter a discount percentage" }]}
      >
        <div className="flex items-center gap-2">
          <Input
            placeholder="5"
            value={newDiscount}
            onChange={(e) => onDiscountChange(e.target.value)}
            className="rounded-lg flex-1"
          />
          <span className="text-muted-foreground">%</span>
        </div>
      </Form.Item>

      <div className="flex items-center justify-end space-x-3 pt-6 border-t border-border">
        <Button
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
