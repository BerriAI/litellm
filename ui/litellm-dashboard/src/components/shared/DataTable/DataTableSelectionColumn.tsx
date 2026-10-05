"use client";

import type { ColumnDef, Row, RowData, Table } from "@tanstack/react-table";
import type React from "react";

import { Checkbox } from "@/components/ui/checkbox";

import { idsBetween, paintRowSelection } from "./rowSelectionRange";

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

const rowIdAt = (tableElement: HTMLTableElement, x: number, y: number): string | null => {
  const rowElement = document.elementFromPoint(x, y)?.closest<HTMLElement>("tr[data-row-id]");
  if (rowElement == null || !tableElement.contains(rowElement)) return null;
  return rowElement.dataset.rowId ?? null;
};

function startDragSelection<TData>(event: React.PointerEvent<HTMLElement>, table: Table<TData>, anchor: Row<TData>) {
  const tableElement = event.currentTarget.closest("table");
  if (tableElement === null) return;
  const rows = table.getRowModel().rows;
  const orderedIds = rows.map((row) => row.id);
  const selectableIds: ReadonlySet<string> = new Set(rows.filter((row) => row.getCanSelect()).map((row) => row.id));
  const snapshot = table.getState().rowSelection;
  const selected = !anchor.getIsSelected();
  const anchorX = event.clientX;

  const paintTo = (rowId: string) => {
    const range = idsBetween(orderedIds, anchor.id, rowId).filter((id) => selectableIds.has(id));
    table.setRowSelection(paintRowSelection(snapshot, range, selected));
  };
  const onMove = (moveEvent: PointerEvent) => {
    const rowId = rowIdAt(tableElement, anchorX, moveEvent.clientY);
    if (rowId !== null) paintTo(rowId);
  };
  const stop = () => {
    document.removeEventListener("pointermove", onMove);
    document.removeEventListener("pointerup", stop);
    document.removeEventListener("pointercancel", stop);
  };

  paintTo(anchor.id);
  document.addEventListener("pointermove", onMove);
  document.addEventListener("pointerup", stop);
  document.addEventListener("pointercancel", stop);
}

function SelectRowCheckbox<TData>({ row, table, label }: { row: Row<TData>; table: Table<TData>; label: string }) {
  const onPointerDown = (event: React.PointerEvent<HTMLElement>) => {
    if (event.button !== 0 || !row.getCanSelect()) return;
    event.preventDefault();
    startDragSelection(event, table, row);
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
