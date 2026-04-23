import React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
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

const InfoTip: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <Info className="ml-2 h-3 w-3 inline text-primary cursor-help" />
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{children}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

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
          <span className="text-sm font-medium text-foreground flex items-center">
            Provider
            <InfoTip>
              Select &apos;Global&apos; to apply margin to all providers, or
              select a specific provider
            </InfoTip>
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
            String(option?.label ?? "")
              .toLowerCase()
              .includes(input.toLowerCase())
          }
        >
          <AntdSelect.Option
            key="global"
            value="global"
            label="Global (All Providers)"
          >
            <div className="flex items-center space-x-2">
              <span className="font-medium">Global (All Providers)</span>
            </div>
          </AntdSelect.Option>
          {Object.entries(Providers).map(
            ([providerEnum, providerDisplayName]) => {
              const providerValue =
                provider_map[providerEnum as keyof typeof provider_map];
              if (providerValue && marginConfig[providerValue]) {
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
            Margin Type
            <InfoTip>
              Choose how to apply the margin: percentage-based or fixed amount
            </InfoTip>
          </span>
        }
        rules={[{ required: true, message: "Please select a margin type" }]}
      >
        <RadioGroup
          value={marginType}
          onValueChange={(v) =>
            onMarginTypeChange(v as "percentage" | "fixed")
          }
          className="flex gap-4"
        >
          <label className="flex items-center gap-2 cursor-pointer">
            <RadioGroupItem value="percentage" />
            Percentage-based
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <RadioGroupItem value="fixed" />
            Fixed Amount
          </label>
        </RadioGroup>
      </Form.Item>

      {marginType === "percentage" && (
        <Form.Item
          label={
            <span className="text-sm font-medium text-foreground flex items-center">
              Margin Percentage
              <InfoTip>
                Enter a percentage value (e.g., 10 for 10% margin)
              </InfoTip>
            </span>
          }
          rules={[
            { required: true, message: "Please enter a margin percentage" },
            {
              validator: (_, value) => {
                if (!value) {
                  return Promise.reject(
                    new Error("Please enter a margin percentage"),
                  );
                }
                const numValue = parseFloat(value);
                if (isNaN(numValue) || numValue < 0 || numValue > 1000) {
                  return Promise.reject(
                    new Error("Percentage must be between 0 and 1000"),
                  );
                }
                return Promise.resolve();
              },
            },
          ]}
        >
          <div className="flex items-center gap-2">
            <Input
              placeholder="10"
              value={percentageValue}
              onChange={(e) => onPercentageChange(e.target.value)}
              className="rounded-lg flex-1"
            />
            <span className="text-muted-foreground">%</span>
          </div>
        </Form.Item>
      )}

      {marginType === "fixed" && (
        <Form.Item
          label={
            <span className="text-sm font-medium text-foreground flex items-center">
              Fixed Margin Amount
              <InfoTip>
                Enter a fixed amount in USD (e.g., 0.001 for $0.001 per
                request)
              </InfoTip>
            </span>
          }
          rules={[
            { required: true, message: "Please enter a fixed amount" },
            {
              validator: (_, value) => {
                if (!value) {
                  return Promise.reject(
                    new Error("Please enter a fixed amount"),
                  );
                }
                const numValue = parseFloat(value);
                if (isNaN(numValue) || numValue < 0) {
                  return Promise.reject(
                    new Error("Fixed amount must be non-negative"),
                  );
                }
                return Promise.resolve();
              },
            },
          ]}
        >
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">$</span>
            <Input
              placeholder="0.001"
              value={fixedAmountValue}
              onChange={(e) => onFixedAmountChange(e.target.value)}
              className="rounded-lg flex-1"
            />
          </div>
        </Form.Item>
      )}

      <div className="flex items-center justify-end space-x-3 pt-6 border-t border-border">
        <Button
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
