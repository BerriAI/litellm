"use client";

import type { ColumnDef, Row, RowData, Table } from "@tanstack/react-table";
import type React from "react";

import { Checkbox } from "@/components/ui/checkbox";

import { startDragSelection } from "./dragRowSelection";

interface SelectionColumnOptions<TData> {
  rowAriaLabel?: (row: Row<TData>) => string;
}

function SelectAllCheckbox<TData>({ table }: { table: Table<TData> }) {
  const allSelected = table.getIsAllPageRowsSelected();
  const someSelected = table.getIsSomePageRowsSelected();

  return (
    <Checkbox
      aria-label="Select all rows"
      data-testid="datatable-select-all"
      checked={allSelected}
      indeterminate={someSelected && !allSelected}
      onCheckedChange={(checked) => table.toggleAllPageRowsSelected(Boolean(checked))}
    />
  );
}

function SelectRowCheckbox<TData>({ row, table, label }: { row: Row<TData>; table: Table<TData>; label: string }) {
  const onPointerDown = (event: React.PointerEvent<HTMLElement>) => {
    const tableElement = event.currentTarget.closest("table");
    if (event.button !== 0 || !row.getCanSelect() || tableElement === null) return;
    event.preventDefault();
    const drag = {
      table,
      anchor: row,
      tableElement,
      pointer: { x: event.clientX, y: event.clientY },
      waitForRowChange: false,
    };
    startDragSelection(drag);
  };
  // Pointer presses are applied on pointerdown so a drag can start there; the click that follows would toggle again.
  const swallowPointerClick = (event: React.MouseEvent<HTMLElement>) => {
    if (event.detail === 0) return;
    event.preventDefault();
    event.stopPropagation();
  };

  return (
    <span className="inline-flex touch-none" onPointerDown={onPointerDown} onClickCapture={swallowPointerClick}>
      <Checkbox
        aria-label={label}
        data-testid={`datatable-select-row-${row.id}`}
        checked={row.getIsSelected()}
        disabled={!row.getCanSelect()}
        onCheckedChange={(checked) => row.toggleSelected(Boolean(checked))}
      />
    </span>
  );
}

export function createSelectionColumn<TData extends RowData>(
  options: SelectionColumnOptions<TData> = {},
): ColumnDef<TData, unknown> {
  const { rowAriaLabel } = options;

  return {
    id: "select",
    size: 44,
    enableSorting: false,
    enableHiding: false,
    enableResizing: false,
    meta: { title: "Select", className: "w-11", headerClassName: "w-11" },
    header: ({ table }) => <SelectAllCheckbox table={table} />,
    cell: ({ row, table }) => <SelectRowCheckbox row={row} table={table} label={rowAriaLabel?.(row) ?? "Select row"} />,
  };
}
