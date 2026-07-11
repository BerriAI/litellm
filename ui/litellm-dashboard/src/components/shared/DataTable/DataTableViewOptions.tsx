"use client";

import { Menu } from "@base-ui/react/menu";
import type { Table } from "@tanstack/react-table";
import { Check, Columns3 } from "lucide-react";

import { Button } from "@/components/ui/button";

interface DataTableViewOptionsProps<TData> {
  table: Table<TData>;
  label?: string;
  className?: string;
}

export function DataTableViewOptions<TData>({ table, label = "View", className }: DataTableViewOptionsProps<TData>) {
  const hideableColumns = table.getAllLeafColumns().filter((column) => column.getCanHide());

  if (hideableColumns.length === 0) {
    return null;
  }

  return (
    <Menu.Root>
      <Menu.Trigger
        render={
          <Button variant="outline" size="sm" className={className} data-testid="view-options-trigger">
            <Columns3 />
            {label}
          </Button>
        }
      />
      <Menu.Portal>
        <Menu.Positioner side="bottom" align="end" sideOffset={4} className="isolate z-50">
          <Menu.Popup className="min-w-[12rem] rounded-md bg-popover p-1 text-sm text-popover-foreground shadow-md ring-1 ring-foreground/10 outline-hidden">
            {hideableColumns.map((column) => (
              <Menu.CheckboxItem
                key={column.id}
                checked={column.getIsVisible()}
                onCheckedChange={(checked) => column.toggleVisibility(checked)}
                closeOnClick={false}
                data-testid={`view-option-${column.id}`}
                className="relative flex cursor-default items-center gap-2 rounded-sm py-1.5 pr-2 pl-7 capitalize outline-hidden select-none data-highlighted:bg-accent data-highlighted:text-accent-foreground"
              >
                <Menu.CheckboxItemIndicator className="absolute left-2 flex size-4 items-center justify-center">
                  <Check className="size-3.5" />
                </Menu.CheckboxItemIndicator>
                {column.columnDef.meta?.title ??
                  (typeof column.columnDef.header === "string" ? column.columnDef.header : column.id)}
              </Menu.CheckboxItem>
            ))}
          </Menu.Popup>
        </Menu.Positioner>
      </Menu.Portal>
    </Menu.Root>
  );
}
