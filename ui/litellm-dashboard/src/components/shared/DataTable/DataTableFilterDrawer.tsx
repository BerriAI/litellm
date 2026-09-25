"use client";

import type { ColumnFiltersState, Table } from "@tanstack/react-table";
import * as React from "react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Sheet, SheetContent, SheetDescription, SheetFooter, SheetHeader, SheetTitle } from "@/components/ui/sheet";

export interface FilterDraft {
  get: (columnId: string) => unknown;
  set: (columnId: string, value: unknown) => void;
}

interface DataTableFilterDrawerProps<TData> {
  table: Table<TData>;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title?: string;
  description?: React.ReactNode;
  applyLabel?: string;
  resetLabel?: string;
  /** Runs instead of the default "clear this table's column filters" when the reset button is pressed. */
  onReset?: () => void;
  children: (draft: FilterDraft) => React.ReactNode;
}

function isEmpty(value: unknown): boolean {
  if (Array.isArray(value)) {
    return value.length === 0;
  }
  return value === undefined || value === null || value === "";
}

function toDraft(filters: ColumnFiltersState): Record<string, unknown> {
  return Object.fromEntries(filters.map((filter) => [filter.id, filter.value]));
}

function toFilters(draft: Record<string, unknown>): ColumnFiltersState {
  return Object.entries(draft)
    .filter(([, value]) => !isEmpty(value))
    .map(([id, value]) => ({ id, value }));
}

export function DataTableFilterDrawer<TData>({
  table,
  open,
  onOpenChange,
  title = "Filters",
  description,
  applyLabel = "Apply Filters",
  resetLabel = "Reset",
  onReset,
  children,
}: DataTableFilterDrawerProps<TData>) {
  const [draft, setDraft] = React.useState<Record<string, unknown>>(() => toDraft(table.getState().columnFilters));
  const [wasOpen, setWasOpen] = React.useState(open);

  if (open !== wasOpen) {
    setWasOpen(open);
    if (open) {
      setDraft(toDraft(table.getState().columnFilters));
    }
  }

  const helpers: FilterDraft = {
    get: (columnId) => draft[columnId],
    set: (columnId, value) => setDraft((previous) => ({ ...previous, [columnId]: value })),
  };

  const apply = () => {
    table.setColumnFilters(toFilters(draft));
    onOpenChange(false);
  };

  const reset = () => {
    setDraft({});
    if (onReset !== undefined) {
      onReset();
      return;
    }
    table.setColumnFilters([]);
  };

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right">
        <SheetHeader>
          <SheetTitle>{title}</SheetTitle>
          {description !== undefined && <SheetDescription>{description}</SheetDescription>}
        </SheetHeader>
        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4" data-testid="filter-drawer-body">
          {children(helpers)}
        </div>
        <SheetFooter className="flex-row">
          <Button variant="outline" className="flex-1" onClick={reset} data-testid="filter-drawer-reset">
            {resetLabel}
          </Button>
          <Button className="flex-1" onClick={apply} data-testid="filter-drawer-apply">
            {applyLabel}
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}

export function DataTableFilterField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label>{label}</Label>
      {children}
    </div>
  );
}
