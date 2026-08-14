"use client";

import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
import type { UIEvent } from "react";

import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";

const SCROLL_THRESHOLD = 0.8;

const SEARCH_REASONS: ReadonlySet<string> = new Set(["input-change", "input-clear", "clear-press"]);

interface UsePaginatedComboboxOptions {
  onSearchChange: (query: string) => void;
  onLoadMore: () => void;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
}

interface PaginatedComboboxHandlers {
  handleSearchInput: (next: string, reason: string) => void;
  handleScroll: (event: UIEvent<HTMLDivElement>) => void;
}

export function usePaginatedCombobox({
  onSearchChange,
  onLoadMore,
  hasNextPage,
  isFetchingNextPage,
}: UsePaginatedComboboxOptions): PaginatedComboboxHandlers {
  const debouncedSearch = useDebouncedCallback(onSearchChange, { wait: DEBOUNCE_WAIT_MS });

  const handleSearchInput = (next: string, reason: string) => {
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

  return { handleSearchInput, handleScroll };
}
