/**
 * Component for configuring a single fallback group
 * Handles primary model selection and fallback chain configuration
 */

import { Select } from "antd";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { AlertCircle, ArrowDown, X } from "lucide-react";
import React from "react";

export interface FallbackGroup {
  id: string;
  primaryModel: string | null;
  fallbackModels: string[];
}

interface FallbackGroupConfigProps {
  group: FallbackGroup;
  onChange: (updatedGroup: FallbackGroup) => void;
  availableModels: string[];
  maxFallbacks: number;
}

export function FallbackGroupConfig({
  group,
  onChange,
  availableModels,
  maxFallbacks,
}: FallbackGroupConfigProps) {
  // Filter available options for fallbacks (exclude primary only, allow already selected to be shown for deselection)
  const availableFallbackOptions = availableModels.filter(
    (m) => m !== group.primaryModel,
  );

  const handlePrimaryChange = (value: string) => {
    let newFallbacks = [...group.fallbackModels];
    // Remove from fallbacks if it was there
    if (newFallbacks.includes(value)) {
      newFallbacks = newFallbacks.filter((m) => m !== value);
    }
    onChange({
      ...group,
      primaryModel: value,
      fallbackModels: newFallbacks,
    });
  };

  const handleFallbackSelect = (values: string[]) => {
    // Limit to maxFallbacks
    const limitedValues = values.slice(0, maxFallbacks);

    onChange({
      ...group,
      fallbackModels: limitedValues,
    });
  };

  const removeFallback = (indexToRemove: number) => {
    const newFallbacks = group.fallbackModels.filter((_, index) => index !== indexToRemove);
    onChange({
      ...group,
      fallbackModels: newFallbacks,
    });
  };

  const canAddMoreFallbacks = group.fallbackModels.length < maxFallbacks;

  return (
    <div className="flex flex-col gap-8 py-4">
      {/* Primary Model Section */}
      <div className="relative">
        <label className="block text-sm font-semibold text-foreground mb-2">
          Primary Model <span className="text-destructive">*</span>
        </label>
        <Select
          className="w-full h-12"
          size="large"
          placeholder="Select primary model"
          value={group.primaryModel}
          onChange={handlePrimaryChange}
          showSearch
          getPopupContainer={(trigger) => trigger.parentElement || document.body}
          filterOption={(input, option) =>
            (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
          }
          options={availableModels.map((m) => ({ label: m, value: m }))}
        />
        {!group.primaryModel && (
          <div className="mt-2 flex items-center gap-2 text-amber-600 dark:text-amber-400 text-xs bg-amber-50 dark:bg-amber-950/30 p-2 rounded">
            <AlertCircle className="w-4 h-4" />
            <span>Select a model to begin configuring fallbacks</span>
          </div>
        )}
      </div>

      {/* Visual Connection */}
      <div className="flex items-center justify-center -my-4 z-10">
        <div className="bg-indigo-50 dark:bg-indigo-950/30 text-indigo-500 dark:text-indigo-400 px-4 py-1 rounded-full text-xs font-bold border border-indigo-100 dark:border-indigo-900 flex items-center gap-2 shadow-sm">
          <ArrowDown className="w-4 h-4" />
          IF FAILS, TRY...
        </div>
      </div>

      {/* Fallback Models Section */}
      <div
        className={`transition-opacity duration-300 ${!group.primaryModel ? "opacity-50 pointer-events-none" : "opacity-100"}`}
      >
        <label className="block text-sm font-semibold text-foreground mb-2">
          Fallback Chain <span className="text-destructive">*</span>
          <span className="text-xs text-muted-foreground font-normal ml-2">
            (Max {maxFallbacks} fallbacks at a time)
          </span>
        </label>

        <div className="bg-muted rounded-xl p-4 border border-border">
          {/* Add Fallback Input */}
          <div className="mb-4">
            <Select
              mode="multiple"
              className="w-full"
              size="large"
              placeholder={
                canAddMoreFallbacks
                  ? "Select fallback models to add..."
                  : `Maximum ${maxFallbacks} fallbacks reached`
              }
              value={group.fallbackModels}
              onChange={handleFallbackSelect}
              disabled={!group.primaryModel}
              getPopupContainer={(trigger) => trigger.parentElement || document.body}
              options={availableFallbackOptions.map((m) => ({
                label: m,
                value: m,
              }))}
              optionRender={(option) => {
                const isSelected = group.fallbackModels.includes(
                  option.value as string,
                );
                const orderIndex = isSelected
                  ? group.fallbackModels.indexOf(option.value as string) + 1
                  : null;
                return (
                  <div className="flex items-center gap-2">
                    {isSelected && orderIndex !== null && (
                      <span className="flex items-center justify-center w-5 h-5 rounded bg-indigo-100 text-indigo-600 dark:bg-indigo-950 dark:text-indigo-300 text-xs font-bold">
                        {orderIndex}
                      </span>
                    )}
                    <span>{option.label}</span>
                  </div>
                );
              }}
              maxTagCount="responsive"
              maxTagPlaceholder={(omittedValues) => (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span>+{omittedValues.length} more</span>
                    </TooltipTrigger>
                    <TooltipContent>
                      {omittedValues.map(({ value }) => value).join(", ")}
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
              showSearch
              filterOption={(input, option) =>
                (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
              }
            />
            <p className="text-xs text-muted-foreground mt-1 ml-1">
              {canAddMoreFallbacks
                ? `Search and select multiple models. Selected models will appear below in order. (${group.fallbackModels.length}/${maxFallbacks} used)`
                : `Maximum ${maxFallbacks} fallbacks reached. Remove some to add more.`}
            </p>
          </div>

          {/* Fallback List */}
          <div className="space-y-2 min-h-[100px]">
            {group.fallbackModels.length === 0 ? (
              <div className="h-32 border-2 border-dashed border-border rounded-lg flex flex-col items-center justify-center text-muted-foreground">
                <span className="text-sm">No fallback models selected</span>
                <span className="text-xs mt-1">
                  Add models from the dropdown above
                </span>
              </div>
            ) : (
              group.fallbackModels.map((modelValue, index) => (
                <div
                  key={`${modelValue}-${index}`}
                  className="group flex items-center justify-between p-3 bg-background rounded-lg border border-border hover:border-indigo-300 dark:hover:border-indigo-700 hover:shadow-sm transition-all"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex items-center justify-center w-6 h-6 rounded bg-muted text-muted-foreground group-hover:text-indigo-500 group-hover:bg-indigo-50 dark:group-hover:bg-indigo-950/30">
                      <span className="text-xs font-bold">{index + 1}</span>
                    </div>
                    <div>
                      <span className="font-medium text-foreground">
                        {modelValue}
                      </span>
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={() => removeFallback(index)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive p-1"
                    aria-label="Remove fallback"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
