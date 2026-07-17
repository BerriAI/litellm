"use client";

import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/cva.config";

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
}

export function DataTablePagination({
  page,
  pageSize,
  rowCount,
  onPageChange,
  onPageSizeChange,
  pageSizeOptions = DEFAULT_PAGE_SIZE_OPTIONS,
  isLoading = false,
  className,
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
        <span>Rows per page</span>
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
          {rowCount === 0 ? "No results" : `Showing ${start}-${end} of ${rowCount}`}
        </span>
        <span data-testid="pagination-page" className="text-sm text-muted-foreground tabular-nums">
          Page {page + 1} of {Math.max(pageCount, 1)}
        </span>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="icon-sm"
            data-testid="pagination-first"
            aria-label="Go to first page"
            disabled={!canPrev}
            onClick={() => onPageChange(0)}
          >
            <ChevronsLeft />
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            data-testid="pagination-prev"
            aria-label="Go to previous page"
            disabled={!canPrev}
            onClick={() => onPageChange(page - 1)}
          >
            <ChevronLeft />
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            data-testid="pagination-next"
            aria-label="Go to next page"
            disabled={!canNext}
            onClick={() => onPageChange(page + 1)}
          >
            <ChevronRight />
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            data-testid="pagination-last"
            aria-label="Go to last page"
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
