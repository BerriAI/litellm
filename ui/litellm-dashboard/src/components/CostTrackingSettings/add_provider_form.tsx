import React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Info } from "lucide-react";
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
  const availableProviders = Object.entries(Providers).filter(([providerEnum]) => {
    const providerValue = provider_map[providerEnum as keyof typeof provider_map];
    return !(providerValue && discountConfig[providerValue]);
  });

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Label htmlFor="discount-provider" className="flex items-center text-sm font-medium text-foreground">
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
        </Label>
        <Select value={selectedProvider} onValueChange={(value) => onProviderChange(value || undefined)}>
          <SelectTrigger id="discount-provider" className="w-full">
            <SelectValue placeholder="Select provider" />
          </SelectTrigger>
          <SelectContent>
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
        <Label htmlFor="discount-percentage" className="flex items-center text-sm font-medium text-foreground">
          Discount Percentage
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="ml-2 h-3 w-3 text-primary cursor-help" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">Enter a percentage value (e.g., 5 for 5% discount)</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </Label>
        <div className="flex items-center gap-2">
          <Input
            id="discount-percentage"
            placeholder="5"
            value={newDiscount}
            onChange={(e) => onDiscountChange(e.target.value)}
            className="rounded-lg flex-1"
          />
          <span className="text-muted-foreground">%</span>
        </div>
      </div>

      <div className="flex items-center justify-end space-x-3 pt-6 border-t border-border">
        <Button onClick={onAddProvider} disabled={!selectedProvider || !newDiscount}>
          Add Provider Discount
        </Button>
      </div>
    </div>
  );
};

export default AddProviderForm;
