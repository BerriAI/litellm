"use client";

import { Check, ChevronsUpDown, Loader2 } from "lucide-react";
import * as React from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/cva.config";

export interface ComboboxOption {
  label: string;
  value: string;
}

interface ComboboxProps {
  options: ComboboxOption[];
  value?: string;
  onValueChange: (value: string) => void;
  placeholder?: string;
  searchPlaceholder?: string;
  emptyText?: string;
  loading?: boolean;
  searchValue?: string;
  onSearchChange?: (value: string) => void;
  onReachEnd?: () => void;
  disabled?: boolean;
  className?: string;
}

const SCROLL_THRESHOLD = 0.8;

export function Combobox({
  options,
  value,
  onValueChange,
  placeholder = "Select…",
  searchPlaceholder = "Search…",
  emptyText = "No results",
  loading = false,
  searchValue,
  onSearchChange,
  onReachEnd,
  disabled = false,
  className,
}: ComboboxProps) {
  const [open, setOpen] = React.useState(false);
  const [internalSearch, setInternalSearch] = React.useState("");
  const isServerSearch = onSearchChange !== undefined;
  const search = isServerSearch ? searchValue ?? "" : internalSearch;

  const handleSearch = (next: string) => {
    if (isServerSearch) {
      onSearchChange(next);
    } else {
      setInternalSearch(next);
    }
  };

  const visibleOptions = isServerSearch
    ? options
    : options.filter((option) => option.label.toLowerCase().includes(internalSearch.trim().toLowerCase()));

  const selectedLabel = options.find((option) => option.value === value)?.label ?? value;

  const handleScroll = (event: React.UIEvent<HTMLDivElement>) => {
    if (onReachEnd === undefined) return;
    const el = event.currentTarget;
    if ((el.scrollTop + el.clientHeight) / el.scrollHeight >= SCROLL_THRESHOLD) {
      onReachEnd();
    }
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        render={
          <Button
            variant="outline"
            size="sm"
            role="combobox"
            aria-expanded={open}
            disabled={disabled}
            className={cn("w-full justify-between gap-2 font-normal", value ? "" : "text-muted-foreground", className)}
          />
        }
      >
        <span className="truncate">{value ? selectedLabel : placeholder}</span>
        <ChevronsUpDown className="size-4 shrink-0 opacity-50" />
      </PopoverTrigger>
      <PopoverContent align="start" className="w-(--anchor-width) gap-0 p-0">
        <div className="border-b border-border p-2">
          <Input
            value={search}
            onChange={(event) => handleSearch(event.target.value)}
            placeholder={searchPlaceholder}
            className="h-8"
            autoFocus
          />
        </div>
        <div className="max-h-56 overflow-y-auto p-1" onScroll={handleScroll} data-testid="combobox-list">
          {value ? (
            <button
              type="button"
              onClick={() => {
                onValueChange("");
                handleSearch("");
                setOpen(false);
              }}
              className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            >
              <span className="size-4 shrink-0" />
              Clear selection
            </button>
          ) : null}
          {visibleOptions.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => {
                onValueChange(option.value);
                setOpen(false);
              }}
              className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-sm hover:bg-accent hover:text-accent-foreground"
            >
              <Check className={cn("size-4 shrink-0", option.value === value ? "opacity-100" : "opacity-0")} />
              <span className="truncate">{option.label}</span>
            </button>
          ))}
          {loading ? (
            <div className="flex items-center justify-center gap-2 py-3 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" /> Loading…
            </div>
          ) : null}
          {!loading && visibleOptions.length === 0 ? (
            <div className="px-2 py-6 text-center text-sm text-muted-foreground">{emptyText}</div>
          ) : null}
        </div>
      </PopoverContent>
    </Popover>
  );
}
