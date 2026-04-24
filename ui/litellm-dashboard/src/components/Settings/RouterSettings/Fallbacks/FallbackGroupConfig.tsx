/**
 * Component for configuring a single fallback group
 * Handles primary model selection and fallback chain configuration
 */

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, ArrowDown, Check, X } from "lucide-react";
import React, { useMemo, useState } from "react";

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

function SearchableMultiSelect({
  options,
  values,
  onChange,
  placeholder,
  disabled,
  maxSelected,
}: {
  options: string[];
  values: string[];
  onChange: (next: string[]) => void;
  placeholder: string;
  disabled?: boolean;
  maxSelected: number;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  const filtered = useMemo(
    () =>
      options.filter((o) =>
        query ? o.toLowerCase().includes(query.toLowerCase()) : true,
      ),
    [options, query],
  );

  const toggle = (value: string) => {
    if (values.includes(value)) {
      onChange(values.filter((v) => v !== value));
    } else if (values.length < maxSelected) {
      onChange([...values, value]);
    }
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className="min-h-10 w-full flex flex-wrap items-center gap-1 rounded-md border border-input bg-background px-2 py-1.5 text-sm text-left disabled:opacity-50"
        >
          {values.length === 0 ? (
            <span className="text-muted-foreground px-1">{placeholder}</span>
          ) : (
            <>
              {values.map((v, i) => (
                <Badge key={v} variant="secondary" className="inline-flex items-center gap-1">
                  <span className="text-xs">{i + 1}.</span>
                  {v}
                  <span
                    role="button"
                    tabIndex={0}
                    aria-label={`Remove ${v}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onChange(values.filter((x) => x !== v));
                    }}
                    className="inline-flex items-center"
                  >
                    <X size={12} />
                  </span>
                </Badge>
              ))}
            </>
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-[var(--radix-popover-trigger-width)] p-2">
        <Input
          autoFocus
          placeholder="Search models…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="h-8 mb-2"
        />
        <div className="max-h-60 overflow-y-auto">
          {filtered.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">No models found</div>
          ) : (
            filtered.map((opt) => {
              const isSelected = values.includes(opt);
              const orderIndex = isSelected ? values.indexOf(opt) + 1 : null;
              return (
                <button
                  key={opt}
                  type="button"
                  onClick={() => toggle(opt)}
                  className="w-full text-left px-2 py-1.5 text-sm rounded hover:bg-accent flex items-center gap-2"
                >
                  {isSelected && orderIndex !== null ? (
                    /* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */
                    <span className="flex items-center justify-center w-5 h-5 rounded bg-indigo-100 text-indigo-600 dark:bg-indigo-950 dark:text-indigo-300 text-xs font-bold">
                      {orderIndex}
                    </span>
                  ) : (
                    <span className="w-5 h-5 inline-flex items-center justify-center">
                      {isSelected ? <Check size={14} /> : null}
                    </span>
                  )}
                  <span>{opt}</span>
                </button>
              );
            })
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}

export function FallbackGroupConfig({
  group,
  onChange,
  availableModels,
  maxFallbacks,
}: FallbackGroupConfigProps) {
  const availableFallbackOptions = availableModels.filter(
    (m) => m !== group.primaryModel,
  );

  const handlePrimaryChange = (value: string) => {
    let newFallbacks = [...group.fallbackModels];
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
          value={group.primaryModel ?? ""}
          onValueChange={handlePrimaryChange}
        >
          <SelectTrigger className="w-full h-12">
            <SelectValue placeholder="Select primary model" />
          </SelectTrigger>
          <SelectContent>
            {availableModels.map((m) => (
              <SelectItem key={m} value={m}>
                {m}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {!group.primaryModel && (
          /* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */
          <div className="mt-2 flex items-center gap-2 text-amber-600 dark:text-amber-400 text-xs bg-amber-50 dark:bg-amber-950/30 p-2 rounded">
            <AlertCircle className="w-4 h-4" />
            <span>Select a model to begin configuring fallbacks</span>
          </div>
        )}
      </div>

      {/* Visual Connection */}
      <div className="flex items-center justify-center -my-4 z-10">
        {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
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
            <SearchableMultiSelect
              options={availableFallbackOptions}
              values={group.fallbackModels}
              onChange={handleFallbackSelect}
              placeholder={
                canAddMoreFallbacks
                  ? "Select fallback models to add..."
                  : `Maximum ${maxFallbacks} fallbacks reached`
              }
              disabled={!group.primaryModel}
              maxSelected={maxFallbacks}
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
                  /* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */
                  className="group flex items-center justify-between p-3 bg-background rounded-lg border border-border hover:border-indigo-300 dark:hover:border-indigo-700 hover:shadow-sm transition-all"
                >
                  <div className="flex items-center gap-3">
                    {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
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
