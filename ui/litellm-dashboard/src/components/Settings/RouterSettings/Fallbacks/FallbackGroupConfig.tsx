/**
 * Component for configuring a single fallback group
 * Handles primary model selection and fallback chain configuration
 */

import { Select, Tooltip } from "antd";
import { AlertCircle, ArrowDown, X } from "lucide-react";
import React from "react";
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
}

export function FallbackGroupConfig({ group, onChange, availableModels, maxFallbacks }: FallbackGroupConfigProps) {
  const { t } = useTranslation();
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

  return (
    <div className="flex flex-col gap-8 py-4">
      {/* Primary Model Section */}
      <div className="relative">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          {t("settingsPages.fallbackGroupConfig.primaryModelLabel")} <span className="text-red-500">*</span>
        </label>
        <Select
          className="w-full h-12"
          size="large"
          placeholder={t("settingsPages.fallbackGroupConfig.selectPrimaryModelPlaceholder")}
          value={group.primaryModel}
          onChange={handlePrimaryChange}
          showSearch
          getPopupContainer={(trigger) => trigger.parentElement || document.body}
          filterOption={(input, option) => (option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
          options={availableModels.map((m) => ({ label: m, value: m }))}
        />
        {!group.primaryModel && (
          <div className="mt-2 flex items-center gap-2 text-amber-600 text-xs bg-amber-50 p-2 rounded">
            <AlertCircle className="w-4 h-4" />
            <span>{t("settingsPages.fallbackGroupConfig.selectModelHint")}</span>
          </div>
        )}
      </div>

      {/* Visual Connection */}
      <div className="flex items-center justify-center -my-4 z-10">
        <div className="bg-indigo-50 text-indigo-500 px-4 py-1 rounded-full text-xs font-bold border border-indigo-100 flex items-center gap-2 shadow-sm">
          <ArrowDown className="w-4 h-4" />
          {t("settingsPages.fallbackGroupConfig.ifFailsTry")}
        </div>
      </div>

      {/* Fallback Models Section */}
      <div
        className={`transition-opacity duration-300 ${!group.primaryModel ? "opacity-50 pointer-events-none" : "opacity-100"}`}
      >
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          {t("settingsPages.fallbackGroupConfig.fallbackChainLabel")} <span className="text-red-500">*</span>
          <span className="text-xs text-gray-500 font-normal ml-2">
            {t("settingsPages.fallbackGroupConfig.maxFallbacksHint", { maxFallbacks })}
          </span>
        </label>

        <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
          {/* Add Fallback Input */}
          <div className="mb-4">
            <Select
              mode="multiple"
              className="w-full"
              size="large"
              placeholder={
                canAddMoreFallbacks
                  ? t("settingsPages.fallbackGroupConfig.selectFallbackPlaceholder")
                  : t("settingsPages.fallbackGroupConfig.maxFallbacksReached", { maxFallbacks })
              }
              value={group.fallbackModels}
              onChange={handleFallbackSelect}
              disabled={!group.primaryModel}
              getPopupContainer={(trigger) => trigger.parentElement || document.body}
              options={availableFallbackOptions.map((m) => ({
                label: m,
                value: m,
              }))}
              optionRender={(option, info) => {
                const isSelected = group.fallbackModels.includes(option.value as string);
                const orderIndex = isSelected ? group.fallbackModels.indexOf(option.value as string) + 1 : null;
                return (
                  <div className="flex items-center gap-2">
                    {isSelected && orderIndex !== null && (
                      <span className="flex items-center justify-center w-5 h-5 rounded bg-indigo-100 text-indigo-600 text-xs font-bold">
                        {orderIndex}
                      </span>
                    )}
                    <span>{option.label}</span>
                  </div>
                );
              }}
              maxTagCount="responsive"
              maxTagPlaceholder={(omittedValues) => (
                <Tooltip
                  styles={{ root: { pointerEvents: "none" } }}
                  title={omittedValues.map(({ value }) => value).join(", ")}
                >
                  <span>+{omittedValues.length} more</span>
                </Tooltip>
              )}
              showSearch
              filterOption={(input, option) => (option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
            />
            <p className="text-xs text-gray-500 mt-1 ml-1">
              {canAddMoreFallbacks
                ? t("settingsPages.fallbackGroupConfig.selectHint", {
                    used: group.fallbackModels.length,
                    max: maxFallbacks,
                  })
                : t("settingsPages.fallbackGroupConfig.maxReachedHint", { maxFallbacks })}
            </p>
          </div>

          {/* Fallback List */}
          <div className="space-y-2 min-h-[100px]">
            {group.fallbackModels.length === 0 ? (
              <div className="h-32 border-2 border-dashed border-gray-300 rounded-lg flex flex-col items-center justify-center text-gray-400">
                <span className="text-sm">{t("settingsPages.fallbackGroupConfig.noFallbacksSelected")}</span>
                <span className="text-xs mt-1">{t("settingsPages.fallbackGroupConfig.addFromDropdown")}</span>
              </div>
            ) : (
              group.fallbackModels.map((modelValue, index) => {
                return (
                  <div
                    key={`${modelValue}-${index}`}
                    className="group flex items-center justify-between p-3 bg-white rounded-lg border border-gray-200 hover:border-indigo-300 hover:shadow-sm transition-all"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex items-center justify-center w-6 h-6 rounded bg-gray-100 text-gray-400 group-hover:text-indigo-500 group-hover:bg-indigo-50">
                        <span className="text-xs font-bold">{index + 1}</span>
                      </div>
                      <div>
                        <span className="font-medium text-gray-800">{modelValue}</span>
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={() => removeFallback(index)}
                      className="opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 hover:text-red-500 p-1"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
