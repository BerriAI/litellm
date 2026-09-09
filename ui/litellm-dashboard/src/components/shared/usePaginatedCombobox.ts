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
  const [typedQuery, setTypedQuery] = useState<string | null>(null);

  const handleInputValueChange = (next: string, reason: string) => {
    if (!SEARCH_REASONS.has(reason)) {
      setTypedQuery(null);
      return;
    }
    setTypedQuery(next);
    debouncedSearch(next);
  };

  const handleOpenChange = (open: boolean, reason: string) => {
    if (!open) {
      if (typedQuery) debouncedSearch("");
      setTypedQuery(null);
      return;
    }
    const openedByTyping = SEARCH_REASONS.has(reason);
    if (!openedByTyping) setTypedQuery("");
  };

  const handleScroll = (event: UIEvent<HTMLDivElement>) => {
    const target = event.currentTarget;
    if (target.scrollHeight === 0) return;
    const ratio = (target.scrollTop + target.clientHeight) / target.scrollHeight;
    if (ratio >= SCROLL_THRESHOLD && hasNextPage && !isFetchingNextPage) {
      onLoadMore?.();
    }
  };

  return { typedQuery, handleInputValueChange, handleOpenChange, handleScroll };
}
