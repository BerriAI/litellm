import React, { useMemo, useState } from "react";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";

interface ModelSelectorProps {
  value: string;
  onChange: (value: string) => void;
  models: string[];
  loading?: boolean;
  disabled?: boolean;
}

interface ModelOption {
  value: string;
  label: string;
}

const CUSTOM_VALUE = "__custom__";

const matchesQuery = (option: ModelOption, query: string): boolean =>
  option.label.toLowerCase().includes(query.trim().toLowerCase());

export function ModelSelector({ value, onChange, models, loading, disabled }: ModelSelectorProps) {
  const [isAddingCustom, setIsAddingCustom] = useState(false);
  const [customValue, setCustomValue] = useState("");

  const options = useMemo(() => Array.from(new Set(models)).sort(), [models]);
  const displayOptions = useMemo(() => {
    if (value && !options.includes(value)) {
      return [value, ...options];
    }
    return options;
  }, [options, value]);

  const modelOptions: ModelOption[] = useMemo(
    () => [
      ...displayOptions.map((model) => ({ value: model, label: model })),
      { value: CUSTOM_VALUE, label: "+ Add custom model" },
    ],
    [displayOptions],
  );

  const selected = isAddingCustom
    ? (modelOptions.find((option) => option.value === CUSTOM_VALUE) ?? null)
    : (modelOptions.find((option) => option.value === value) ?? null);

  const handleSelect = (option: ModelOption | null) => {
    if (option?.value === CUSTOM_VALUE) {
      setIsAddingCustom(true);
      setCustomValue(value && !options.includes(value) ? value : "");
      return;
    }
    setIsAddingCustom(false);
    setCustomValue("");
    onChange(option?.value ?? "");
  };

  const commitCustomValue = () => {
    const trimmed = customValue.trim();
    if (!trimmed) {
      setIsAddingCustom(false);
      setCustomValue("");
      return;
    }
    onChange(trimmed);
    setIsAddingCustom(false);
    setCustomValue("");
  };

  return (
    <div className="min-w-0 flex-1">
      <Combobox
        items={modelOptions}
        value={selected}
        onValueChange={handleSelect}
        disabled={disabled}
        isItemEqualToValue={(a: ModelOption, b: ModelOption) => a.value === b.value}
        itemToStringLabel={(option: ModelOption) => option.label}
        filter={matchesQuery}
      >
        <ComboboxInput
          placeholder={loading ? "Loading models..." : "Select a model"}
          className="w-full"
          disabled={disabled}
        />
        <ComboboxContent>
          <ComboboxEmpty>
            {loading ? (
              <span aria-busy="true" className="flex items-center justify-center py-2">
                <UiLoadingSpinner className="size-4" />
              </span>
            ) : (
              "No models available"
            )}
          </ComboboxEmpty>
          <ComboboxList>
            {(option: ModelOption) => (
              <ComboboxItem key={option.value} value={option}>
                {option.label}
              </ComboboxItem>
            )}
          </ComboboxList>
        </ComboboxContent>
      </Combobox>
      {isAddingCustom && (
        <Input
          className="mt-2"
          placeholder="Custom Model Name (Enter to add)"
          value={customValue}
          autoFocus
          onChange={(event) => setCustomValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              commitCustomValue();
            }
          }}
          onBlur={commitCustomValue}
        />
      )}
    </div>
  );
}
