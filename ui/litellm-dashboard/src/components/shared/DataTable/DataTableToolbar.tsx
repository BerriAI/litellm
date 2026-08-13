"use client";

import type { Table } from "@tanstack/react-table";
import { RefreshCw, Search, SlidersHorizontal, X } from "lucide-react";
import type * as React from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/cva.config";

import { DataTableViewOptions } from "./DataTableViewOptions";

interface DataTableToolbarProps<TData> {
  table: Table<TData>;
  searchValue?: string;
  onSearchChange?: (value: string) => void;
  searchPlaceholder?: string;
  onOpenFilters?: () => void;
  onRefresh?: () => void;
  isRefreshing?: boolean;
  filterLabels?: Record<string, string>;
  formatFilterValue?: (columnId: string, value: unknown) => string;
  showViewOptions?: boolean;
  children?: React.ReactNode;
  className?: string;
}

function defaultFormatValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  return String(value);
}

export function DataTableToolbar<TData>({
  table,
  searchValue,
  onSearchChange,
  searchPlaceholder = "Search",
  onOpenFilters,
  onRefresh,
  isRefreshing = false,
  filterLabels,
  formatFilterValue,
  showViewOptions = true,
  children,
  className,
}: DataTableToolbarProps<TData>) {
  const filters = table.getState().columnFilters;

  const labelFor = (columnId: string): string =>
    filterLabels?.[columnId] ?? table.getColumn(columnId)?.columnDef.meta?.title ?? columnId;
  const valueFor = (columnId: string, value: unknown): string =>
    formatFilterValue?.(columnId, value) ?? defaultFormatValue(value);

  return (
    <div className={cn("flex flex-wrap items-center justify-between gap-2", className)}>
      <div className="flex flex-1 flex-wrap items-center gap-2">
        {onSearchChange !== undefined && (
          <div className="relative">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={searchValue ?? ""}
              onChange={(event) => onSearchChange(event.target.value)}
              placeholder={searchPlaceholder}
              className="h-8 w-56 pl-8"
              data-testid="datatable-search"
            />
          </div>
        )}
        {filters.map((filter) => (
          <Badge key={filter.id} variant="outline" className="gap-1 py-1" data-testid={`filter-chip-${filter.id}`}>
            <span className="text-muted-foreground">{labelFor(filter.id)}:</span>
            {valueFor(filter.id, filter.value)}
            <button
              type="button"
              aria-label={`Remove ${labelFor(filter.id)} filter`}
              data-testid={`filter-chip-remove-${filter.id}`}
              onClick={() => table.setColumnFilters((previous) => previous.filter((entry) => entry.id !== filter.id))}
              className="ml-0.5 rounded-full text-muted-foreground hover:text-foreground"
            >
              <X className="size-3" />
            </button>
          </Badge>
        ))}
        {filters.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => table.setColumnFilters([])}
            data-testid="datatable-clear-filters"
          >
            Clear all
          </Button>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {children}
        {onRefresh !== undefined && (
          <Button
            variant="outline"
            size="icon-sm"
            onClick={onRefresh}
            disabled={isRefreshing}
            aria-label="Refresh"
            title="Refresh"
            data-testid="datatable-refresh"
          >
            <RefreshCw className={isRefreshing ? "animate-spin" : ""} />
          </Button>
        )}
        {showViewOptions && <DataTableViewOptions table={table} label="Columns" />}
        {onOpenFilters !== undefined && (
          <Button variant="outline" size="sm" onClick={onOpenFilters} data-testid="datatable-filters-trigger">
            <SlidersHorizontal />
            Filters
            {filters.length > 0 && (
              <Badge className="ml-1 h-5 min-w-5 justify-center rounded-full px-1" data-testid="datatable-filter-count">
                {filters.length}
              </Badge>
            )}
          </Button>
        )}
      </div>
    </div>
  );
}
