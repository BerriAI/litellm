"use client";

import { Search } from "lucide-react";
import type * as React from "react";

import { FilterInput } from "@/components/common_components/Filters/FilterInput";
import { FiltersButton } from "@/components/common_components/Filters/FiltersButton";
import { ResetFiltersButton } from "@/components/common_components/Filters/ResetFiltersButton";
import { cn } from "@/lib/cva.config";

interface DataTableToolbarProps {
  searchValue?: string;
  onSearchChange?: (value: string) => void;
  searchPlaceholder?: string;
  filtersActive?: boolean;
  hasActiveFilters?: boolean;
  onToggleFilters?: () => void;
  onResetFilters?: () => void;
  children?: React.ReactNode;
  className?: string;
}

export function DataTableToolbar({
  searchValue,
  onSearchChange,
  searchPlaceholder = "Search",
  filtersActive = false,
  hasActiveFilters = false,
  onToggleFilters,
  onResetFilters,
  children,
  className,
}: DataTableToolbarProps) {
  const showReset = onResetFilters !== undefined && hasActiveFilters;

  return (
    <div className={cn("flex flex-wrap items-center justify-between gap-2 pb-3", className)}>
      <div className="flex flex-wrap items-center gap-2">
        {onSearchChange !== undefined && (
          <FilterInput
            value={searchValue ?? ""}
            onChange={onSearchChange}
            placeholder={searchPlaceholder}
            icon={Search}
          />
        )}
        {onToggleFilters !== undefined && (
          <FiltersButton onClick={onToggleFilters} active={filtersActive} hasActiveFilters={hasActiveFilters} />
        )}
        {showReset && <ResetFiltersButton onClick={onResetFilters} />}
      </div>
      {children !== undefined && <div className="flex flex-wrap items-center gap-2">{children}</div>}
    </div>
  );
}
