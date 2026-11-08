"use client";

import type { ColumnDef, Row, RowData, Table } from "@tanstack/react-table";
import { useTranslation } from "react-i18next";

import { Checkbox } from "@/components/ui/checkbox";
import i18n from "@/i18n/i18n";

interface SelectionColumnOptions<TData> {
  rowAriaLabel?: (row: Row<TData>) => string;
}

function SelectAllCheckbox<TData>({ table }: { table: Table<TData> }) {
  const { t } = useTranslation();
  const allSelected = table.getIsAllPageRowsSelected();
  const someSelected = table.getIsSomePageRowsSelected();

  return (
    <Checkbox
      aria-label={t("shared.dataTable.selectionColumn.selectAllRows")}
      data-testid="datatable-select-all"
      checked={allSelected}
      indeterminate={someSelected && !allSelected}
      onCheckedChange={(checked) => table.toggleAllPageRowsSelected(Boolean(checked))}
    />
  );
}

function SelectRowCheckbox<TData>({ row, label }: { row: Row<TData>; label: string }) {
  return (
    <Checkbox
      aria-label={label}
      data-testid={`datatable-select-row-${row.id}`}
      checked={row.getIsSelected()}
      disabled={!row.getCanSelect()}
      onCheckedChange={(checked) => row.toggleSelected(Boolean(checked))}
    />
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
    cell: ({ row }) => (
      <SelectRowCheckbox
        row={row}
        label={rowAriaLabel?.(row) ?? i18n.t("shared.dataTable.selectionColumn.selectRow")}
      />
    ),
  };
}
