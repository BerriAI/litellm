"use client";

import { Loader2 } from "lucide-react";
import { useMemo, useState } from "react";

import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";

import type { SearchSelectOption } from "./SearchSelect";
import { usePaginatedCombobox } from "./usePaginatedCombobox";

interface PaginatedSearchSelectProps {
  options: SearchSelectOption[];
  value?: string;
  onValueChange: (value: string) => void;
  onSearchChange: (query: string) => void;
  onLoadMore?: () => void;
  hasNextPage?: boolean;
  isLoading?: boolean;
  isFetchingNextPage?: boolean;
  placeholder?: string;
  emptyText?: string;
  errorText?: string;
  loadingText?: string;
  disabled?: boolean;
  className?: string;
  inputId?: string;
  "aria-invalid"?: true | undefined;
  "aria-describedby"?: string;
}

export function PaginatedSearchSelect({
  options,
  value,
  onValueChange,
  onSearchChange,
  onLoadMore,
  hasNextPage = false,
  isLoading = false,
  isFetchingNextPage = false,
  placeholder = "Search…",
  emptyText = "No results",
  errorText,
  loadingText = "Loading…",
  disabled = false,
  className,
  inputId,
  "aria-invalid": ariaInvalid,
  "aria-describedby": ariaDescribedBy,
}: PaginatedSearchSelectProps) {
  const [pickedOption, setPickedOption] = useState<SearchSelectOption | null>(null);

  const selected = useMemo<SearchSelectOption | null>(() => {
    if (value === undefined || value === "") return null;
    return (
      options.find((option) => option.value === value) ??
      (pickedOption?.value === value ? pickedOption : { label: value, value })
    );
  }, [options, value, pickedOption]);

  const items = useMemo<SearchSelectOption[]>(() => {
    if (selected === null) return options;
    if (options.some((option) => option.value === selected.value)) return options;
    return [selected, ...options];
  }, [options, selected]);

  const pagination = { onSearchChange, onLoadMore, hasNextPage, isFetchingNextPage };
  const { query, handleInputValueChange, handleOpenChange, handleScroll } = usePaginatedCombobox(pagination);

  return (
    <Combobox
      items={items}
      value={selected}
      inputValue={query ?? selected?.label ?? ""}
      onValueChange={(item: SearchSelectOption | null) => {
        setPickedOption(item);
        onValueChange(item?.value ?? "");
      }}
      onInputValueChange={(next, eventDetails) => handleInputValueChange(next, eventDetails.reason)}
      onOpenChange={(nextOpen, eventDetails) => handleOpenChange(nextOpen, eventDetails.reason)}
      isItemEqualToValue={(a: SearchSelectOption, b: SearchSelectOption) => a.value === b.value}
      itemToStringLabel={(item: SearchSelectOption) => item.label}
      filter={null}
      disabled={disabled}
    >
      <ComboboxInput
        id={inputId}
        aria-invalid={ariaInvalid}
        aria-describedby={ariaDescribedBy}
        placeholder={placeholder}
        showClear={value !== undefined && value !== ""}
        className={`w-full ${className ?? ""}`}
      />
      <ComboboxContent>
        <ComboboxEmpty className={errorText == null ? undefined : "text-destructive"}>
          {errorText ?? (isLoading ? loadingText : emptyText)}
        </ComboboxEmpty>
        <ComboboxList onScroll={handleScroll} data-testid="paginated-search-select-list">
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
        {isFetchingNextPage && (
          <div className="flex justify-center py-2" data-testid="paginated-search-select-loading-more">
            <Loader2 className="size-4 animate-spin text-muted-foreground" />
          </div>
        )}
      </ComboboxContent>
    </Combobox>
  );
}
