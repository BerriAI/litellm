"use client";

import { Menu } from "@base-ui/react/menu";
import type { Column, SortDirection } from "@tanstack/react-table";
import { ChevronDown, ChevronsUpDown, ChevronUp, X } from "lucide-react";
import type * as React from "react";

import { cn } from "@/lib/cva.config";

export type DataTableSortVariant = "header-cycle" | "dropdown-tristate";

interface DataTableSortHeaderProps<TData, TValue> {
  column: Column<TData, TValue>;
  title: React.ReactNode;
  variant?: DataTableSortVariant;
  className?: string;
}

function SortIndicator({ sorted }: { sorted: false | SortDirection }) {
  if (sorted === "asc") {
    return <ChevronUp className="size-3.5" data-sort-indicator="asc" />;
  }
  if (sorted === "desc") {
    return <ChevronDown className="size-3.5" data-sort-indicator="desc" />;
  }
  return <ChevronsUpDown className="size-3.5 text-muted-foreground" data-sort-indicator="none" />;
}

const MENU_ITEM_CLASS =
  "flex cursor-default items-center gap-2 rounded-sm px-2 py-1.5 outline-hidden select-none data-highlighted:bg-accent data-highlighted:text-accent-foreground";

export function DataTableSortHeader<TData, TValue>({
  column,
  title,
  variant = "header-cycle",
  className,
}: DataTableSortHeaderProps<TData, TValue>) {
  const sorted = column.getIsSorted();

  if (!column.getCanSort()) {
    return <span className={cn("font-medium", className)}>{title}</span>;
  }

  if (variant === "dropdown-tristate") {
    return (
      <div className={cn("flex items-center gap-1", className)}>
        <span className="font-medium">{title}</span>
        <Menu.Root>
          <Menu.Trigger
            render={
              <button
                type="button"
                data-testid={`sort-trigger-${column.id}`}
                aria-label={`Sort options for ${column.id}`}
                onClick={(event) => event.stopPropagation()}
                className={cn(
                  "inline-flex size-6 items-center justify-center rounded-md hover:bg-muted",
                  sorted ? "text-primary" : "text-muted-foreground",
                )}
              >
                <SortIndicator sorted={sorted} />
              </button>
            }
          />
          <Menu.Portal>
            <Menu.Positioner side="bottom" align="start" sideOffset={4} className="isolate z-50">
              <Menu.Popup className="min-w-[9rem] rounded-md bg-popover p-1 text-sm text-popover-foreground shadow-md ring-1 ring-foreground/10 outline-hidden">
                <Menu.Item className={MENU_ITEM_CLASS} onClick={() => column.toggleSorting(false)}>
                  <ChevronUp className="size-3.5" /> Ascending
                </Menu.Item>
                <Menu.Item className={MENU_ITEM_CLASS} onClick={() => column.toggleSorting(true)}>
                  <ChevronDown className="size-3.5" /> Descending
                </Menu.Item>
                <Menu.Item className={MENU_ITEM_CLASS} onClick={() => column.clearSorting()}>
                  <X className="size-3.5" /> Reset
                </Menu.Item>
              </Menu.Popup>
            </Menu.Positioner>
          </Menu.Portal>
        </Menu.Root>
      </div>
    );
  }

  return (
    <button
      type="button"
      data-testid={`sort-header-${column.id}`}
      onClick={column.getToggleSortingHandler()}
      className={cn("flex items-center gap-1 font-medium select-none hover:text-foreground", className)}
    >
      <span>{title}</span>
      <SortIndicator sorted={sorted} />
    </button>
  );
}
