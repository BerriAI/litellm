import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Info } from "lucide-react";
import React, { useEffect, useState } from "react";

interface AdditionalModelSettingsProps {
  temperature?: number;
  maxTokens?: number;
  useAdvancedParams?: boolean;
  onTemperatureChange?: (value: number) => void;
  onMaxTokensChange?: (value: number) => void;
  onUseAdvancedParamsChange?: (value: boolean) => void;
  mockTestFallbacks?: boolean;
  onMockTestFallbacksChange?: (value: boolean) => void;
}

const AdditionalModelSettings: React.FC<AdditionalModelSettingsProps> = ({
  temperature = 1.0,
  maxTokens = 2048,
  useAdvancedParams: externalUseAdvancedParams,
  onTemperatureChange,
  onMaxTokensChange,
  onUseAdvancedParamsChange,
  mockTestFallbacks,
  onMockTestFallbacksChange,
}) => {
  const [internalUseAdvancedParams, setInternalUseAdvancedParams] =
    useState(false);
  const useAdvancedParams =
    externalUseAdvancedParams !== undefined
      ? externalUseAdvancedParams
      : internalUseAdvancedParams;
  const [localTemperature, setLocalTemperature] = useState(temperature);
  const [localMaxTokens, setLocalMaxTokens] = useState(maxTokens);

  useEffect(() => {
    setLocalTemperature(temperature);
  }, [temperature]);

  useEffect(() => {
    setLocalMaxTokens(maxTokens);
  }, [maxTokens]);

  const handleTemperatureChange = (value: number) => {
    setLocalTemperature(value);
    onTemperatureChange?.(value);
  };

  const handleMaxTokensChange = (value: number) => {
    setLocalMaxTokens(value);
    onMaxTokensChange?.(value);
  };

  const handleUseAdvancedParamsChange = (checked: boolean) => {
    if (onUseAdvancedParamsChange) {
      onUseAdvancedParamsChange(checked);
    } else {
      setInternalUseAdvancedParams(checked);
    }
  };

  const disabledTextColor = useAdvancedParams
    ? "text-foreground"
    : "text-muted-foreground";

  return (
    <div className="space-y-4 p-4 w-80">
      <label className="flex items-center gap-2 cursor-pointer">
        <Checkbox
          checked={useAdvancedParams}
          onCheckedChange={(c) => handleUseAdvancedParamsChange(c === true)}
        />
        <span className="font-medium">Use Advanced Parameters</span>
      </label>

      {onMockTestFallbacksChange && (
        <div className="flex items-center gap-1">
          <label className="flex items-center gap-2 cursor-pointer">
            <Checkbox
              checked={mockTestFallbacks ?? false}
              onCheckedChange={(c) =>
                onMockTestFallbacksChange(c === true)
              }
            />
            <span className="font-medium">
              Simulate failure to test fallbacks
            </span>
          </label>
          <Popover>
            <PopoverTrigger asChild>
              <button
                type="button"
                className="text-xs text-muted-foreground cursor-pointer shrink-0 hover:text-foreground"
                aria-label="Help: Simulate failure to test fallbacks"
              >
                <Info className="h-3 w-3" />
              </button>
            </PopoverTrigger>
            <PopoverContent className="max-w-[340px]">
              <p className="text-sm mb-2">
                Causes the first request to fail so the router tries
                fallbacks (if configured). Use this to verify your fallback
                setup.
              </p>
              <p className="text-sm">
                Behavior can differ when keys, teams, or router settings are
                configured.{" "}
                <a
                  href="https://docs.litellm.ai/docs/proxy/keys_teams_router_settings"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  Learn more
                </a>
              </p>
            </PopoverContent>
          </Popover>
        </div>
      )}

      <div
        className={cn(
          "space-y-4 transition-opacity duration-200",
          !useAdvancedParams && "opacity-40",
        )}
      >
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1">
              <span className={cn("text-sm", disabledTextColor)}>
                Temperature
              </span>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Info
                      className={cn(
                        "h-3 w-3 cursor-help",
                        disabledTextColor,
                      )}
                    />
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    Controls randomness. Lower values make output more
                    deterministic, higher values more creative.
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            <Input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={localTemperature}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                handleTemperatureChange(isNaN(v) ? 1.0 : v);
              }}
              disabled={!useAdvancedParams}
              className="w-20 h-8"
            />
          </div>
          <Slider
            min={0}
            max={2}
            step={0.1}
            value={[localTemperature]}
            onValueChange={([v]) => handleTemperatureChange(v)}
            disabled={!useAdvancedParams}
          />
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>0</span>
            <span>1.0</span>
            <span>2.0</span>
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1">
              <span className={cn("text-sm", disabledTextColor)}>
                Max Tokens
              </span>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Info
                      className={cn(
                        "h-3 w-3 cursor-help",
                        disabledTextColor,
                      )}
                    />
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    Maximum number of tokens to generate in the response.
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            <Input
              type="number"
              min={1}
              max={32768}
              step={1}
              value={localMaxTokens}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                handleMaxTokensChange(isNaN(v) ? 1000 : v);
              }}
              disabled={!useAdvancedParams}
              className="w-24 h-8"
            />
          </div>
          <Slider
            min={1}
            max={32768}
            step={1}
            value={[localMaxTokens]}
            onValueChange={([v]) => handleMaxTokensChange(v)}
            disabled={!useAdvancedParams}
          />
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>1</span>
            <span>32768</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdditionalModelSettings;
