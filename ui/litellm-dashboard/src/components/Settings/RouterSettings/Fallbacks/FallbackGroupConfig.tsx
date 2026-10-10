/**
 * Component for configuring a single fallback group
 * Handles primary model selection and fallback chain configuration
 */

import { MultiSelect } from "@/components/shared/MultiSelect";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { AlertCircle, ArrowDown, X } from "lucide-react";
import React, { useId } from "react";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { toast } from "@/lib/toast";
import { primaryModels, providerModels } from "./fallbackModels";

export interface FallbackGroup {
  id: string;
  primaryModel: string | null;
  primaryProvider?: string;
  fallbackModels: string[];
}

interface FallbackGroupConfigProps {
  group: FallbackGroup;
  onChange: (updatedGroup: FallbackGroup) => void;
  availableModels: string[];
  modelInfo?: ModelGroup[];
  maxFallbacks: number;
  disablePrimaryModel?: boolean;
}

export function FallbackGroupConfig({
  group,
  onChange,
  availableModels,
  modelInfo = [],
  maxFallbacks,
  disablePrimaryModel = false,
}: FallbackGroupConfigProps) {
  const primaries = primaryModels(group, modelInfo);
  const hasPrimary = primaries.length > 0;
  const providers = [...new Set(modelInfo.flatMap((model) => model.providers ?? []))].sort();
  const modelIcon = (model: string) => (
    <span className="inline-flex shrink-0 gap-1">
      {(modelInfo.find((info) => info.model_group === model)?.providers ?? []).map((provider) => (
        <ProviderLogo key={provider} provider={provider} />
      ))}
    </span>
  );
  const modelNames = [
    ...new Set([...availableModels, ...group.fallbackModels, ...(group.primaryModel ? [group.primaryModel] : [])]),
  ];
  const modelOptions = modelNames.map((model) => ({
    value: JSON.stringify(["model", model]),
    label: model,
    models: [model],
    provider: undefined as string | undefined,
    icon: modelIcon(model),
  }));
  const providerOptions = providers
    .map((provider) => ({
      value: JSON.stringify(["provider", provider]),
      label: `All ${getProviderLogoAndName(provider).displayName} Models`,
      models: providerModels(modelInfo, provider),
      provider,
      icon: <ProviderLogo provider={provider} />,
    }))
    .filter((option) => option.models.length > 0);
  const primaryOptions = [...providerOptions, ...modelOptions];
  const fallbackOptions = primaryOptions
    .map((option) => ({
      ...option,
      models: option.models.filter((model) => !primaries.includes(model)),
    }))
    .filter((option) => option.models.length > 0);

  const handlePrimaryChange = (value: string | null) => {
    const selected = primaryOptions.find((option) => option.value === value);
    const newFallbacks = group.fallbackModels.filter((model) => !selected?.models.includes(model));
    const updatedGroup = {
      ...group,
      primaryModel: selected?.provider ? null : selected?.models[0] ?? null,
      primaryProvider: selected?.provider,
      fallbackModels: newFallbacks,
    };
    onChange(updatedGroup);
  };

  const handleFallbackSelect = (values: string[]) => {
    const selected = [
      ...new Set(
        values.flatMap(
          (value) =>
            fallbackOptions.find((option) => option.value === value)?.models ??
            group.fallbackModels.filter((model) => JSON.stringify(["model", model]) === value),
        ),
      ),
    ];
    if (selected.length > maxFallbacks) {
      toast.error(`This selection contains ${selected.length} models. Choose at most ${maxFallbacks}.`);
      return;
    }

    onChange({
      ...group,
      fallbackModels: selected,
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
  const fallbackInputId = useId();

  const moveFallback = (index: number, direction: number) => {
    const target = index + direction;
    onChange({
      ...group,
      fallbackModels: group.fallbackModels.map((model, position) => {
        if (position === target) return group.fallbackModels[index];
        return position === index ? group.fallbackModels[target] : model;
      }),
    });
  };
  const selectedModelValue = group.primaryModel ? JSON.stringify(["model", group.primaryModel]) : null;
  const selectedPrimaryValue = group.primaryProvider
    ? JSON.stringify(["provider", group.primaryProvider])
    : selectedModelValue;

  return (
    <div className="flex flex-col gap-8 py-4">
      {/* Primary Model Section */}
      <div className="relative">
        <label htmlFor={primaryModelInputId} className="block text-sm font-semibold text-foreground mb-2">
          Primary Model <span className="text-destructive">*</span>
        </label>
        <SearchSelect
          inputId={primaryModelInputId}
          options={
            disablePrimaryModel
              ? modelOptions
              : primaryOptions.map((option) => ({
                  ...option,
                  sublabel: option.provider ? `${option.models.length} currently configured models` : undefined,
                }))
          }
          value={selectedPrimaryValue}
          onValueChange={handlePrimaryChange}
          placeholder="Select primary model"
          emptyText="No models found"
          disabled={disablePrimaryModel}
          className="h-12"
        />
        {group.primaryProvider && (
          <div className="mt-3 space-y-2 text-sm">
            <p>
              Applies to {primaries.length} currently configured models. Newly added models and wildcard routes are not
              included.
            </p>
            <ul aria-label="Primary models" className="flex flex-wrap gap-2">
              {primaries.map((model) => (
                <li key={model} className="inline-flex items-center gap-2 rounded border px-2 py-1">
                  {modelIcon(model)}
                  {model}
                </li>
              ))}
            </ul>
          </div>
        )}
        {!disablePrimaryModel && !hasPrimary && (
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
      <div className={`transition-opacity duration-300 ${!hasPrimary ? "opacity-50" : "opacity-100"}`}>
        <label htmlFor={fallbackInputId} className="block text-sm font-semibold text-foreground mb-2">
          Fallback Chain <span className="text-destructive">*</span>
          <span className="text-xs text-muted-foreground font-normal ml-2">
            (Max {maxFallbacks} fallbacks at a time)
          </span>
        </label>

        <div className="bg-muted rounded-xl p-4 border border-border">
          {/* Add Fallback Input */}
          <div className="mb-4">
            <MultiSelect
              id={fallbackInputId}
              options={fallbackOptions.map((option) => ({
                ...option,
                description: option.provider
                  ? `${option.models.length} configured models, added alphabetically`
                  : undefined,
              }))}
              value={group.fallbackModels.map((model) => JSON.stringify(["model", model]))}
              onValueChange={handleFallbackSelect}
              placeholder={
                canAddMoreFallbacks ? "Select fallback models to add..." : `Maximum ${maxFallbacks} fallbacks reached`
              }
              emptyText="No models found"
              disabled={!hasPrimary}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground mt-1 ml-1">
              {canAddMoreFallbacks
                ? `Select models or a provider. Review and reorder the chain below. (${group.fallbackModels.length}/${maxFallbacks} used)`
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
                        <span className="inline-flex items-center gap-2 font-medium text-foreground">
                          {modelIcon(modelValue)}
                          {modelValue}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center gap-1">
                      <button
                        type="button"
                        disabled={index === 0}
                        aria-label={`Move ${modelValue} earlier`}
                        onClick={() => moveFallback(index, -1)}
                        className="p-1 disabled:opacity-30"
                      >
                        <ArrowDown className="h-4 w-4 rotate-180" />
                      </button>
                      <button
                        type="button"
                        disabled={index === group.fallbackModels.length - 1}
                        aria-label={`Move ${modelValue} later`}
                        onClick={() => moveFallback(index, 1)}
                        className="p-1 disabled:opacity-30"
                      >
                        <ArrowDown className="h-4 w-4" />
                      </button>
                      <button
                        type="button"
                        aria-label={`Remove ${modelValue}`}
                        onClick={() => removeFallback(index)}
                        className="text-muted-foreground hover:text-destructive p-1"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
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
