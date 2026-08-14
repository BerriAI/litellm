"use client";

import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";

export interface SearchSelectOption {
  label: string;
  value: string;
  /** Optional muted second line (e.g. an id); also matched when searching. */
  sublabel?: string;
}

interface SearchSelectProps {
  options: SearchSelectOption[];
  value?: string;
  onValueChange: (value: string) => void;
  placeholder?: string;
  emptyText?: string;
  disabled?: boolean;
  className?: string;
  inputId?: string;
}

const matchesQuery = (option: SearchSelectOption, query: string): boolean => {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return option.label.toLowerCase().includes(q) || (option.sublabel?.toLowerCase().includes(q) ?? false);
};

export function SearchSelect({
  options,
  value,
  onValueChange,
  placeholder = "Select…",
  emptyText = "No results",
  disabled = false,
  className,
  inputId,
}: SearchSelectProps) {
  const selected =
    value === undefined || value === ""
      ? null
      : options.find((option) => option.value === value) ?? { label: value, value };
  const items =
    selected !== null && !options.some((option) => option.value === selected.value) ? [selected, ...options] : options;

  return (
    <Combobox
      items={items}
      value={selected}
      onValueChange={(item: SearchSelectOption | null) => onValueChange(item?.value ?? "")}
      isItemEqualToValue={(a: SearchSelectOption, b: SearchSelectOption) => a.value === b.value}
      itemToStringLabel={(item: SearchSelectOption) => item.label}
      filter={matchesQuery}
      disabled={disabled}
    >
      <ComboboxInput
        id={inputId}
        placeholder={placeholder}
        showClear={value != null && value !== ""}
        className={`h-8 w-full text-sm ${className ?? ""}`}
      />
      <ComboboxContent side="bottom" collisionAvoidance={{ side: "shift", align: "shift", fallbackAxisSide: "none" }}>
        <ComboboxEmpty>{emptyText}</ComboboxEmpty>
        <ComboboxList>
          {(item: SearchSelectOption) => (
            <ComboboxItem key={item.value} value={item}>
              <span className="flex min-w-0 flex-col">
                <span className="truncate">{item.label}</span>
                {item.sublabel != null && item.sublabel !== "" && (
                  <span className="truncate text-xs text-muted-foreground">{item.sublabel}</span>
                )}
              </span>
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
}
