"use client";

import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
import { Loader2 } from "lucide-react";
import { useMemo, type UIEvent } from "react";

import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";

import type { SearchSelectOption } from "./SearchSelect";

const SCROLL_THRESHOLD = 0.8;

const SEARCH_REASONS: ReadonlySet<string> = new Set(["input-change", "input-clear", "clear-press"]);

interface PaginatedSearchSelectProps {
  options: SearchSelectOption[];
  value?: string;
  onValueChange: (value: string) => void;
  onSearchChange: (query: string) => void;
  onLoadMore: () => void;
  hasNextPage?: boolean;
  isLoading?: boolean;
  isFetchingNextPage?: boolean;
  placeholder?: string;
  emptyText?: string;
  loadingText?: string;
  disabled?: boolean;
  className?: string;
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
  loadingText = "Loading…",
  disabled = false,
  className,
}: PaginatedSearchSelectProps) {
  const selected = useMemo<SearchSelectOption | null>(() => {
    if (value === undefined || value === "") return null;
    return options.find((option) => option.value === value) ?? { label: value, value };
  }, [options, value]);

  const items = useMemo<SearchSelectOption[]>(() => {
    if (selected === null) return options;
    if (options.some((option) => option.value === selected.value)) return options;
    return [selected, ...options];
  }, [options, selected]);

  const debouncedSearch = useDebouncedCallback(onSearchChange, { wait: DEBOUNCE_WAIT_MS });

  const handleInputValueChange = (next: string, reason: string) => {
    if (!SEARCH_REASONS.has(reason)) return;
    debouncedSearch(next);
  };

  const handleScroll = (event: UIEvent<HTMLDivElement>) => {
    const target = event.currentTarget;
    if (target.scrollHeight === 0) return;
    const ratio = (target.scrollTop + target.clientHeight) / target.scrollHeight;
    if (ratio >= SCROLL_THRESHOLD && hasNextPage && !isFetchingNextPage) {
      onLoadMore();
    }
  };

  return (
    <Combobox
      items={items}
      value={selected}
      onValueChange={(item: SearchSelectOption | null) => onValueChange(item?.value ?? "")}
      onInputValueChange={(next, eventDetails) => handleInputValueChange(next, eventDetails.reason)}
      isItemEqualToValue={(a: SearchSelectOption, b: SearchSelectOption) => a.value === b.value}
      itemToStringLabel={(item: SearchSelectOption) => item.label}
      filter={null}
      disabled={disabled}
    >
      <ComboboxInput
        placeholder={placeholder}
        showClear={value !== undefined && value !== ""}
        className={`w-full ${className ?? ""}`}
      />
      <ComboboxContent>
        <ComboboxEmpty>{isLoading ? loadingText : emptyText}</ComboboxEmpty>
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
