"use client";

import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
import { useState, type UIEvent } from "react";

import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";

const SCROLL_THRESHOLD = 0.8;

const SEARCH_REASONS: ReadonlySet<string> = new Set(["input-change", "input-clear", "clear-press"]);

export interface PaginatedComboboxCallbacks {
  onSearchChange: (query: string) => void;
  onLoadMore?: () => void;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
}

export function usePaginatedCombobox({
  onSearchChange,
  onLoadMore,
  hasNextPage,
  isFetchingNextPage,
}: PaginatedComboboxCallbacks) {
  const debouncedSearch = useDebouncedCallback(onSearchChange, { wait: DEBOUNCE_WAIT_MS });
  // null means "not searching": the input then shows the selected option's label instead.
  const [query, setQuery] = useState<string | null>(null);

  const handleInputValueChange = (next: string, reason: string) => {
    if (!SEARCH_REASONS.has(reason)) {
      setQuery(null);
      return;
    }
    setQuery(next);
    debouncedSearch(next);
  };

  const handleOpenChange = (open: boolean, reason: string) => {
    if (!open) {
      if (query) debouncedSearch("");
      setQuery(null);
      return;
    }
    // Typing into a closed combobox opens it, and that keystroke is already the query.
    if (!SEARCH_REASONS.has(reason)) setQuery("");
  };

  const handleScroll = (event: UIEvent<HTMLDivElement>) => {
    const target = event.currentTarget;
    if (target.scrollHeight === 0) return;
    const ratio = (target.scrollTop + target.clientHeight) / target.scrollHeight;
    if (ratio >= SCROLL_THRESHOLD && hasNextPage && !isFetchingNextPage) {
      onLoadMore?.();
    }
  };

  return { query, handleInputValueChange, handleOpenChange, handleScroll };
}
