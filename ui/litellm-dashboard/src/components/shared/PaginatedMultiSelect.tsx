"use client";

import { Loader2 } from "lucide-react";
import { useMemo, useState } from "react";

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
  useComboboxAnchor,
} from "@/components/ui/combobox";

import type { SearchSelectOption } from "./SearchSelect";
import { usePaginatedCombobox } from "./usePaginatedCombobox";

interface PaginatedMultiSelectProps {
  options: SearchSelectOption[];
  value?: string[];
  onValueChange: (value: string[]) => void;
  onSearchChange: (query: string) => void;
  onLoadMore: () => void;
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

export function PaginatedMultiSelect({
  options,
  value = [],
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
}: PaginatedMultiSelectProps) {
  const anchor = useComboboxAnchor();
  const [query, setQuery] = useState("");
  const [pickedOptions, setPickedOptions] = useState<ReadonlyMap<string, SearchSelectOption>>(new Map());

  const selected = useMemo<SearchSelectOption[]>(
    () =>
      value.map(
        (selectedValue) =>
          options.find((option) => option.value === selectedValue) ??
          pickedOptions.get(selectedValue) ?? { label: selectedValue, value: selectedValue },
      ),
    [options, value, pickedOptions],
  );

  const items = useMemo<SearchSelectOption[]>(() => {
    const missing = selected.filter((option) => !options.some((o) => o.value === option.value));
    return missing.length === 0 ? options : [...missing, ...options];
  }, [options, selected]);

  const pagination = { onSearchChange, onLoadMore, hasNextPage, isFetchingNextPage };
  const { handleInputValueChange, handleScroll } = usePaginatedCombobox(pagination);

  const handleChipsInputChange = (next: string, reason: string) => {
    setQuery(next);
    handleInputValueChange(next, reason);
  };

  return (
    <Combobox
      multiple
      items={items}
      value={selected}
      onValueChange={(next: SearchSelectOption[]) => {
        setPickedOptions(new Map(next.map((option) => [option.value, option])));
        onValueChange(next.map((option) => option.value));
      }}
      inputValue={query}
      onInputValueChange={(next, eventDetails) => handleChipsInputChange(next, eventDetails.reason)}
      isItemEqualToValue={(a: SearchSelectOption, b: SearchSelectOption) => a.value === b.value}
      itemToStringLabel={(item: SearchSelectOption) => item.label}
      filter={null}
      disabled={disabled}
    >
      <ComboboxChips render={<div ref={anchor} />} className={`min-h-8 py-1 text-sm ${className ?? ""}`}>
        <ComboboxValue>
          {(selectedItems: SearchSelectOption[]) =>
            selectedItems.map((option) => (
              <ComboboxChip key={option.value} aria-label={option.label}>
                {option.label}
              </ComboboxChip>
            ))
          }
        </ComboboxValue>
        <ComboboxChipsInput
          id={inputId}
          aria-invalid={ariaInvalid}
          aria-describedby={ariaDescribedBy}
          placeholder={placeholder}
          className="h-5 min-w-24 flex-1 border-0 bg-transparent py-0 text-sm"
          aria-label={placeholder}
        />
      </ComboboxChips>
      <ComboboxContent anchor={anchor}>
        <ComboboxEmpty className={errorText == null ? undefined : "text-destructive"}>
          {errorText ?? (isLoading ? loadingText : emptyText)}
        </ComboboxEmpty>
        <ComboboxList onScroll={handleScroll} data-testid="paginated-multi-select-list">
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
          <div className="flex justify-center py-2" data-testid="paginated-multi-select-loading-more">
            <Loader2 className="size-4 animate-spin text-muted-foreground" />
          </div>
        )}
      </ComboboxContent>
    </Combobox>
  );
}
