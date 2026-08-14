"use client";

import { useState } from "react";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
} from "@/components/ui/combobox";

export interface MultiSelectOption {
  label: string;
  value: string;
  description?: string;
}

interface MultiSelectProps {
  options: MultiSelectOption[];
  value?: string[];
  onValueChange: (value: string[]) => void;
  placeholder?: string;
  emptyText?: string;
  disabled?: boolean;
  loading?: boolean;
  allowCustomValues?: boolean;
  className?: string;
}

const matchesQuery = (option: MultiSelectOption, query: string): boolean => {
  const normalizedQuery = query.trim().toLowerCase();
  return (
    !normalizedQuery ||
    option.label.toLowerCase().includes(normalizedQuery) ||
    option.value.toLowerCase().includes(normalizedQuery) ||
    (option.description?.toLowerCase().includes(normalizedQuery) ?? false)
  );
};

export function MultiSelect({
  options,
  value = [],
  onValueChange,
  placeholder = "Select options",
  emptyText = "No options found",
  disabled = false,
  loading = false,
  allowCustomValues = false,
  className,
}: MultiSelectProps) {
  const [query, setQuery] = useState("");
  const safeOptions = options.filter(
    (option): option is MultiSelectOption =>
      option != null && typeof option.value === "string" && option.value.length > 0,
  );
  const selectedOptions = value
    .filter((selectedValue): selectedValue is string => typeof selectedValue === "string" && selectedValue.length > 0)
    .map(
      (selectedValue) =>
        safeOptions.find((option) => option.value === selectedValue) ?? {
          label: selectedValue,
          value: selectedValue,
        },
    );
  const customOption = query.trim();
  const customOptionExists = safeOptions.some((option) => option.value.toLowerCase() === customOption.toLowerCase());
  const items =
    allowCustomValues && customOption && !customOptionExists
      ? [...safeOptions, { label: `Create "${customOption}"`, value: customOption }]
      : safeOptions;

  return (
    <Combobox
      multiple
      items={items}
      value={selectedOptions}
      onValueChange={(selected: MultiSelectOption[]) => {
        onValueChange(selected.map((option) => option.value));
        setQuery("");
      }}
      inputValue={query}
      onInputValueChange={setQuery}
      isItemEqualToValue={(option: MultiSelectOption, selected: MultiSelectOption) => option.value === selected.value}
      itemToStringLabel={(option: MultiSelectOption) => option.label}
      filter={matchesQuery}
      disabled={disabled || loading}
    >
      <ComboboxChips className={`min-h-8 py-1 text-sm ${className ?? ""}`}>
        <ComboboxValue>
          {(selected: MultiSelectOption[]) =>
            selected.map((option) => (
              <ComboboxChip key={option.value} aria-label={option.label}>
                {option.label}
              </ComboboxChip>
            ))
          }
        </ComboboxValue>
        <ComboboxChipsInput
          placeholder={loading ? "Loading..." : placeholder}
          className="h-5 min-w-24 flex-1 border-0 bg-transparent py-0 text-sm"
          aria-label={placeholder}
        />
      </ComboboxChips>
      <ComboboxContent>
        <ComboboxEmpty>{emptyText}</ComboboxEmpty>
        <ComboboxList>
          {(option: MultiSelectOption) => (
            <ComboboxItem key={option.value} value={option}>
              <span className="min-w-0">
                <span className="block truncate">{option.label}</span>
                {option.description && (
                  <span className="block truncate text-xs text-muted-foreground">{option.description}</span>
                )}
              </span>
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
}
