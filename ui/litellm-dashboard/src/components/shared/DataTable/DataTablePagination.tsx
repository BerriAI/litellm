"use client";

import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/cva.config";
import type { DataTablePaginationLabels } from "./types";

export const DEFAULT_PAGE_SIZE_OPTIONS = [25, 50, 100];

export interface DataTablePaginationProps {
  page: number;
  pageSize: number;
  rowCount: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (pageSize: number) => void;
  pageSizeOptions?: number[];
  isLoading?: boolean;
  className?: string;
  labels?: DataTablePaginationLabels;
}

const DEFAULT_LABELS: DataTablePaginationLabels = {
  rowsPerPage: "Rows per page",
  noResults: "No results",
  range: (start, end, total) => `Showing ${start}-${end} of ${total}`,
  page: (current, total) => `Page ${current} of ${total}`,
  firstPage: "Go to first page",
  previousPage: "Go to previous page",
  nextPage: "Go to next page",
  lastPage: "Go to last page",
};

export function DataTablePagination({
  page,
  pageSize,
  rowCount,
  onPageChange,
  onPageSizeChange,
  pageSizeOptions = DEFAULT_PAGE_SIZE_OPTIONS,
  isLoading = false,
  className,
  labels = DEFAULT_LABELS,
}: DataTablePaginationProps) {
  const pageCount = pageSize > 0 ? Math.ceil(rowCount / pageSize) : 0;
  const start = rowCount === 0 ? 0 : page * pageSize + 1;
  const end = Math.min((page + 1) * pageSize, rowCount);
  const canPrev = page > 0 && !isLoading;
  const canNext = page < pageCount - 1 && !isLoading;
  const lastPage = Math.max(pageCount - 1, 0);

  return (
    <div className={cn("flex flex-wrap items-center justify-between gap-4 px-4 py-2.5", className)}>
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <span>{labels.rowsPerPage}</span>
        <Select
          value={String(pageSize)}
          onValueChange={(value) => {
            if (typeof value === "string") {
              onPageSizeChange(Number(value));
            }
          }}
        >
          <SelectTrigger size="sm" data-testid="pagination-page-size" className="w-[4.5rem]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {pageSizeOptions.map((option) => (
              <SelectItem key={option} value={String(option)}>
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex items-center gap-4">
        <span data-testid="pagination-range" className="text-sm text-muted-foreground tabular-nums">
          {rowCount === 0 ? labels.noResults : labels.range(start, end, rowCount)}
        </span>
        <span data-testid="pagination-page" className="text-sm text-muted-foreground tabular-nums">
          {labels.page(page + 1, Math.max(pageCount, 1))}
        </span>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="icon-sm"
            data-testid="pagination-first"
            aria-label={labels.firstPage}
            disabled={!canPrev}
            onClick={() => onPageChange(0)}
          >
            <ChevronsLeft />
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            data-testid="pagination-prev"
            aria-label={labels.previousPage}
            disabled={!canPrev}
            onClick={() => onPageChange(page - 1)}
          >
            <ChevronLeft />
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            data-testid="pagination-next"
            aria-label={labels.nextPage}
            disabled={!canNext}
            onClick={() => onPageChange(page + 1)}
          >
            <ChevronRight />
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            data-testid="pagination-last"
            aria-label={labels.lastPage}
            disabled={!canNext}
            onClick={() => onPageChange(lastPage)}
          >
            <ChevronsRight />
          </Button>
        </div>
      </div>
    </div>
  );
}
