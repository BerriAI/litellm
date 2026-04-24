import React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Info } from "lucide-react";
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
  const availableProviders = Object.entries(Providers).filter(([providerEnum]) => {
    const providerValue = provider_map[providerEnum as keyof typeof provider_map];
    return !(providerValue && marginConfig[providerValue]);
  });

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Label htmlFor="margin-provider" className="flex items-center text-sm font-medium text-foreground">
          Provider
          <InfoTip>Select &apos;Global&apos; to apply margin to all providers, or select a specific provider</InfoTip>
        </Label>
        <Select value={selectedProvider} onValueChange={(value) => onProviderChange(value || undefined)}>
          <SelectTrigger id="margin-provider" className="w-full">
            <SelectValue placeholder="Select provider or 'Global'" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="global">
              <div className="flex items-center space-x-2">
                <span className="font-medium">Global (All Providers)</span>
              </div>
            </SelectItem>
            {availableProviders.map(([providerEnum, providerDisplayName]) => (
              <SelectItem key={providerEnum} value={providerEnum}>
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
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label className="flex items-center text-sm font-medium text-foreground">
          Margin Type
          <InfoTip>Choose how to apply the margin: percentage-based or fixed amount</InfoTip>
        </Label>
        <RadioGroup
          value={marginType}
          onValueChange={(v) => onMarginTypeChange(v as "percentage" | "fixed")}
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
      </div>

      {marginType === "percentage" && (
        <div className="space-y-2">
          <Label htmlFor="margin-percentage" className="flex items-center text-sm font-medium text-foreground">
            Margin Percentage
            <InfoTip>Enter a percentage value (e.g., 10 for 10% margin)</InfoTip>
          </Label>
          <div className="flex items-center gap-2">
            <Input
              id="margin-percentage"
              placeholder="10"
              value={percentageValue}
              onChange={(e) => onPercentageChange(e.target.value)}
              className="rounded-lg flex-1"
            />
            <span className="text-muted-foreground">%</span>
          </div>
        </div>
      )}

      {marginType === "fixed" && (
        <div className="space-y-2">
          <Label htmlFor="margin-fixed" className="flex items-center text-sm font-medium text-foreground">
            Fixed Margin Amount
            <InfoTip>Enter a fixed amount in USD (e.g., 0.001 for $0.001 per request)</InfoTip>
          </Label>
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">$</span>
            <Input
              id="margin-fixed"
              placeholder="0.001"
              value={fixedAmountValue}
              onChange={(e) => onFixedAmountChange(e.target.value)}
              className="rounded-lg flex-1"
            />
          </div>
        </div>
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
