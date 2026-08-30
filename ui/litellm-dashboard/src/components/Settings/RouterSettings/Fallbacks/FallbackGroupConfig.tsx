/**
 * Component for configuring a single fallback group
 * Handles primary model selection and fallback chain configuration
 */

import { MultiSelect } from "@/components/shared/MultiSelect";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { AlertCircle, ArrowDown, X } from "lucide-react";
import React, { useId } from "react";

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
  disablePrimaryModel?: boolean;
}

export function FallbackGroupConfig({
  group,
  onChange,
  availableModels,
  maxFallbacks,
  disablePrimaryModel = false,
}: FallbackGroupConfigProps) {
  // Filter available options for fallbacks (exclude primary only, allow already selected to be shown for deselection)
  const availableFallbackOptions = availableModels.filter((m) => m !== group.primaryModel);

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
  const primaryModelInputId = useId();

  return (
    <div className="flex flex-col gap-8 py-4">
      {/* Primary Model Section */}
      <div className="relative">
        <label htmlFor={primaryModelInputId} className="block text-sm font-semibold text-foreground mb-2">
          Primary Model <span className="text-destructive">*</span>
        </label>
        <SearchSelect
          inputId={primaryModelInputId}
          options={availableModels.map((m) => ({ label: m, value: m }))}
          value={group.primaryModel ?? ""}
          onValueChange={handlePrimaryChange}
          placeholder="Select primary model"
          emptyText="No models found"
          disabled={disablePrimaryModel}
          className="h-12"
        />
        {!disablePrimaryModel && !group.primaryModel && (
          <div className="mt-2 flex items-center gap-2 text-warning text-xs bg-warning/10 p-2 rounded-sm">
            <AlertCircle className="w-4 h-4" />
            <span>Select a model to begin configuring fallbacks</span>
          </div>
        )}
      </div>

      {/* Visual Connection */}
      <div className="flex items-center justify-center -my-4 z-raised">
        <div className="bg-indigo-50 text-indigo-500 px-4 py-1 rounded-full text-xs font-bold border border-indigo-100 flex items-center gap-2 shadow-xs dark:bg-indigo-950 dark:text-indigo-300 dark:border-indigo-900">
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
            <MultiSelect
              options={availableFallbackOptions.map((m) => ({ label: m, value: m }))}
              value={group.fallbackModels}
              onValueChange={handleFallbackSelect}
              placeholder={
                canAddMoreFallbacks ? "Select fallback models to add..." : `Maximum ${maxFallbacks} fallbacks reached`
              }
              emptyText="No models found"
              disabled={!group.primaryModel}
              className="w-full"
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
                <span className="text-xs mt-1">Add models from the dropdown above</span>
              </div>
            ) : (
              <ol aria-label="Fallback chain" className="space-y-2">
                {group.fallbackModels.map((modelValue, index) => (
                  <li
                    key={`${modelValue}-${index}`}
                    className="group flex items-center justify-between p-3 bg-card rounded-lg border border-border hover:border-indigo-300 hover:shadow-xs transition-all"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex items-center justify-center w-6 h-6 rounded-sm bg-muted text-muted-foreground group-hover:text-indigo-500 group-hover:bg-indigo-50 dark:group-hover:text-indigo-300 dark:group-hover:bg-indigo-950">
                        <span className="text-xs font-bold">{index + 1}</span>
                      </div>
                      <div>
                        <span className="font-medium text-foreground">{modelValue}</span>
                      </div>
                    </div>

                    <button
                      type="button"
                      aria-label={`Remove ${modelValue}`}
                      onClick={() => removeFallback(index)}
                      className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive p-1"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </li>
                ))}
              </ol>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
