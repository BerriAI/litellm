"use client";

import { Loader2 } from "lucide-react";
import { useMemo, useRef, useState, type SyntheticEvent } from "react";

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
  autoHighlight?: boolean | "always";
  disabled?: boolean;
  className?: string;
  inputId?: string;
  "aria-required"?: true | undefined;
  "aria-invalid"?: true | undefined;
  "aria-describedby"?: string;
}

const typedInsertion = (previous: string, next: string): string => {
  let start = 0;
  while (start < previous.length && start < next.length && previous[start] === next[start]) start++;
  let end = 0;
  while (
    end < previous.length - start &&
    end < next.length - start &&
    previous[previous.length - 1 - end] === next[next.length - 1 - end]
  )
    end++;
  return next.slice(start, next.length - end);
};

const editedQuery = (label: string, next: string): string => {
  const inserted = typedInsertion(label, next);
  return inserted === "" && next !== label ? next : inserted;
};

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
  autoHighlight = false,
  disabled = false,
  className,
  inputId,
  "aria-required": ariaRequired,
  "aria-invalid": ariaInvalid,
  "aria-describedby": ariaDescribedBy,
}: PaginatedSearchSelectProps) {
  const [pickedOption, setPickedOption] = useState<SearchSelectOption | null>(null);
  const wholeSelectionRef = useRef(false);

  const snapshotWholeSelection = (event: SyntheticEvent<HTMLInputElement>) => {
    const input = event.currentTarget;
    wholeSelectionRef.current =
      input.value.length > 0 && input.selectionStart === 0 && input.selectionEnd === input.value.length;
  };

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
  const { typedQuery, handleInputValueChange, handleOpenChange, handleScroll } = usePaginatedCombobox(pagination);

  const handleTypedInput = (next: string, reason: string) => {
    const replacedWholeInput = wholeSelectionRef.current;
    wholeSelectionRef.current = false;
    handleInputValueChange(
      typedQuery === null && !replacedWholeInput ? editedQuery(selected?.label ?? "", next) : next,
      reason,
    );
  };

  return (
    <Combobox
      items={items}
      value={selected}
      inputValue={typedQuery ?? selected?.label ?? ""}
      onValueChange={(item: SearchSelectOption | null) => {
        setPickedOption(item);
        onValueChange(item?.value ?? "");
      }}
      onInputValueChange={(next, eventDetails) => handleTypedInput(next, eventDetails.reason)}
      onOpenChange={(nextOpen, eventDetails) => handleOpenChange(nextOpen, eventDetails.reason)}
      isItemEqualToValue={(a: SearchSelectOption, b: SearchSelectOption) => a.value === b.value}
      itemToStringLabel={(item: SearchSelectOption) => item.label}
      // @ts-expect-error TS2322 -- Combobox.Root narrows autoHighlight to boolean; the AriaCombobox it wraps
      // accepts "always", the only value that highlights a list filtered server-side
      autoHighlight={autoHighlight}
      filter={null}
      disabled={disabled}
    >
      <ComboboxInput
        id={inputId}
        aria-required={ariaRequired}
        aria-invalid={ariaInvalid}
        aria-describedby={ariaDescribedBy}
        onFocus={(event) => event.currentTarget.select()}
        onKeyDown={snapshotWholeSelection}
        onPaste={snapshotWholeSelection}
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
