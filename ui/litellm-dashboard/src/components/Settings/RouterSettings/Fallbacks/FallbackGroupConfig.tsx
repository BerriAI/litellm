/**
 * Component for configuring a single fallback group
 * Handles primary model selection and fallback chain configuration
 */

import { MultiSelect } from "@/components/shared/MultiSelect";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { AlertCircle, ArrowDown, X } from "lucide-react";
import React, { useId } from "react";
import { useTranslation } from "react-i18next";

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
  const { t } = useTranslation();
  // Filter available options for fallbacks (exclude primary only, allow already selected to be shown for deselection)
  const availableFallbackOptions = availableModels.filter((m) => m !== group.primaryModel);

  const handlePrimaryChange = (value: string | null) => {
    const newFallbacks = group.fallbackModels.filter((model) => model !== value);
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
          {t("routerSettings.fallbackGroupConfig.primaryModel")} <span className="text-destructive">*</span>
        </label>
        <SearchSelect
          inputId={primaryModelInputId}
          options={availableModels.map((m) => ({ label: m, value: m }))}
          value={group.primaryModel}
          onValueChange={handlePrimaryChange}
          placeholder={t("routerSettings.fallbackGroupConfig.primaryPlaceholder")}
          emptyText={t("routerSettings.fallbackGroupConfig.noModelsFound")}
          disabled={disablePrimaryModel}
          className="h-12"
        />
        {!disablePrimaryModel && !group.primaryModel && (
          <div className="mt-2 flex items-center gap-2 text-warning text-xs bg-warning/10 p-2 rounded-sm">
            <AlertCircle className="w-4 h-4" />
            <span>{t("routerSettings.fallbackGroupConfig.selectPrimaryHint")}</span>
          </div>
        )}
      </div>

      {/* Visual Connection */}
      <div className="flex items-center justify-center -my-4 z-raised">
        <div className="bg-indigo-50 text-indigo-500 px-4 py-1 rounded-full text-xs font-bold border border-indigo-100 flex items-center gap-2 shadow-xs dark:bg-indigo-950 dark:text-indigo-300 dark:border-indigo-900">
          <ArrowDown className="w-4 h-4" />
          {t("routerSettings.fallbackGroupConfig.ifFails")}
        </div>
      </div>

      {/* Fallback Models Section */}
      <div
        className={`transition-opacity duration-300 ${!group.primaryModel ? "opacity-50 pointer-events-none" : "opacity-100"}`}
      >
        <label className="block text-sm font-semibold text-foreground mb-2">
          {t("routerSettings.fallbackGroupConfig.fallbackChain")} <span className="text-destructive">*</span>
          <span className="text-xs text-muted-foreground font-normal ml-2">
            {t("routerSettings.fallbackGroupConfig.maxAtATime", { max: maxFallbacks })}
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
                canAddMoreFallbacks
                  ? t("routerSettings.fallbackGroupConfig.addPlaceholder")
                  : t("routerSettings.fallbackGroupConfig.maxReached", { max: maxFallbacks })
              }
              emptyText={t("routerSettings.fallbackGroupConfig.noModelsFound")}
              disabled={!group.primaryModel}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground mt-1 ml-1">
              {canAddMoreFallbacks
                ? t("routerSettings.fallbackGroupConfig.searchHint", {
                    used: group.fallbackModels.length,
                    max: maxFallbacks,
                  })
                : t("routerSettings.fallbackGroupConfig.maxReachedHint", { max: maxFallbacks })}
            </p>
          </div>

          {/* Fallback List */}
          <div className="space-y-2 min-h-[100px]">
            {group.fallbackModels.length === 0 ? (
              <div className="h-32 border-2 border-dashed border-border rounded-lg flex flex-col items-center justify-center text-muted-foreground">
                <span className="text-sm">{t("routerSettings.fallbackGroupConfig.noFallbacksSelected")}</span>
                <span className="text-xs mt-1">{t("routerSettings.fallbackGroupConfig.addFromDropdown")}</span>
              </div>
            ) : (
              <ol aria-label={t("routerSettings.fallbackGroupConfig.chainAriaLabel")} className="space-y-2">
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
                      aria-label={t("routerSettings.fallbackGroupConfig.removeModel", { model: modelValue })}
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
