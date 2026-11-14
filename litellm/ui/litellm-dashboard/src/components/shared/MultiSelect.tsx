"use client";

import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
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
import i18n from "@/i18n/i18n";

export interface MultiSelectOption {
  label: string;
  value: string;
  description?: string;
  disabled?: boolean;
  icon?: ReactNode;
}

interface MultiSelectProps {
  id?: string;
  "aria-describedby"?: string;
  options: MultiSelectOption[];
  value?: string[];
  onValueChange: (value: string[]) => void;
  placeholder?: string;
  emptyText?: string;
  disabled?: boolean;
  loading?: boolean;
  allowCustomValues?: boolean;
  className?: string;
  contentClassName?: string;
}

const splitOnCommas = (raw: string): string[] =>
  raw
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0);

const matchesQuery = (option: MultiSelectOption, query: string): boolean => {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) return true;
  if (option.label.toLowerCase().includes(normalizedQuery)) return true;
  if (option.value.toLowerCase().includes(normalizedQuery)) return true;
  return option.description?.toLowerCase().includes(normalizedQuery) ?? false;
};

export function MultiSelect({
  id,
  "aria-describedby": describedBy,
  options,
  value = [],
  onValueChange,
  placeholder = i18n.t("shared.multiSelect.placeholder"),
  emptyText = i18n.t("shared.multiSelect.empty"),
  disabled = false,
  loading = false,
  allowCustomValues = false,
  className,
  contentClassName,
}: MultiSelectProps) {
  const { t } = useTranslation();
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
      ? [...safeOptions, { label: t("shared.multiSelect.createOption", { value: customOption }), value: customOption }]
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
                  {option.icon && (
                    <span aria-hidden="true" className="shrink-0">
                      {option.icon}
                    </span>
                  )}
                  {option.label}
                </ComboboxChip>
              ))}
              <ComboboxChipsInput
                id={id}
                aria-describedby={describedBy}
                placeholder={loading ? t("shared.multiSelect.loading") : placeholder}
                className="min-w-24"
                aria-label={placeholder || undefined}
              />
              {canClear(selected) && (
                <ComboboxClear className="ml-auto self-center" aria-label={t("shared.multiSelect.clearAll")} />
              )}
            </>
          )}
        </ComboboxValue>
      </ComboboxChips>
      <ComboboxContent anchor={anchor} className={contentClassName}>
        <ComboboxEmpty>{emptyText}</ComboboxEmpty>
        <ComboboxList>
          {(option: MultiSelectOption) => (
            <ComboboxItem key={option.value} value={option} disabled={option.disabled}>
              <span className="flex min-w-0 items-center gap-2">
                {option.icon && (
                  <span aria-hidden="true" className="shrink-0">
                    {option.icon}
                  </span>
                )}
                <span className="min-w-0">
                  <span className="block truncate">{option.label}</span>
                  {option.description && (
                    <span className="block truncate text-xs text-muted-foreground">{option.description}</span>
                  )}
                </span>
              </span>
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
}
