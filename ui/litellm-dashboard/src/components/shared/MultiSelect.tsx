"use client";

import { useState } from "react";
import {
  Combobox,
  ComboboxChip,
  ComboboxClear,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
} from "@/components/ui/combobox";

export interface MultiSelectOption {
  label: string;
  value: string;
  description?: string;
  disabled?: boolean;
}

interface MultiSelectProps {
  id?: string;
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

const splitOnCommas = (raw: string): string[] =>
  raw
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0);

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
  id,
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
  const anchor = useComboboxAnchor();
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

  const canClear = (selected: MultiSelectOption[]) => selected.length > 0 && !disabled && !loading;

  const handleValueChange = (selected: MultiSelectOption[]) => {
    const next = allowCustomValues
      ? selected.flatMap((option) => (value.includes(option.value) ? [option.value] : splitOnCommas(option.value)))
      : selected.map((option) => option.value);
    onValueChange(Array.from(new Set(next)));
    setQuery("");
  };

  return (
    <Combobox
      multiple
      items={items}
      value={selectedOptions}
      onValueChange={handleValueChange}
      inputValue={query}
      onInputValueChange={setQuery}
      isItemEqualToValue={(option: MultiSelectOption, selected: MultiSelectOption) => option.value === selected.value}
      itemToStringLabel={(option: MultiSelectOption) => option.label}
      filter={matchesQuery}
      disabled={disabled || loading}
    >
      <ComboboxChips render={<div ref={anchor} />} className={`min-h-8 py-1 text-sm ${className ?? ""}`}>
        <ComboboxValue>
          {(selected: MultiSelectOption[]) => (
            <>
              {selected.map((option) => (
                <ComboboxChip key={option.value} aria-label={option.label}>
                  {option.label}
                </ComboboxChip>
              ))}
              <ComboboxChipsInput
                id={id}
                placeholder={loading ? "Loading..." : placeholder}
                className="min-w-24"
                aria-label={placeholder || undefined}
              />
              {canClear(selected) && <ComboboxClear className="ml-auto self-center" aria-label="Clear all" />}
            </>
          )}
        </ComboboxValue>
      </ComboboxChips>
      <ComboboxContent anchor={anchor}>
        <ComboboxEmpty>{emptyText}</ComboboxEmpty>
        <ComboboxList>
          {(option: MultiSelectOption) => (
            <ComboboxItem key={option.value} value={option} disabled={option.disabled}>
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
