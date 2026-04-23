import React from "react";
// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import {
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
} from "@tremor/react";

export interface SimpleTableColumn<T> {
  header: string;
  accessor?: keyof T;
  cell?: (row: T) => React.ReactNode;
  width?: string;
}

interface SimpleTableProps<T> {
  data: T[];
  columns: SimpleTableColumn<T>[];
  isLoading?: boolean;
  loadingMessage?: string;
  emptyMessage?: string;
  getRowKey?: (row: T, index: number) => string;
}

/**
 * Simple table component for forms and settings pages.
 * The underlying primitives are still Tremor's Table family (whitelisted
 * by the banned-imports rule); only the Text wrapper was swapped for a
 * semantic <span>.
 */
export function SimpleTable<T>({
  data,
  columns,
  isLoading = false,
  loadingMessage = "Loading...",
  emptyMessage = "No data",
  getRowKey,
}: SimpleTableProps<T>) {
  return (
    <Table>
      <TableHead>
        <TableRow>
          {columns.map((column, index) => (
            <TableHeaderCell key={index} style={{ width: column.width }}>
              {column.header}
            </TableHeaderCell>
          ))}
        </TableRow>
      </TableHead>
      <TableBody>
        {isLoading ? (
          <TableRow>
            <TableCell colSpan={columns.length} className="text-center">
              <span className="text-muted-foreground">{loadingMessage}</span>
            </TableCell>
          </TableRow>
        ) : data.length > 0 ? (
          data.map((row, rowIndex) => (
            <TableRow key={getRowKey ? getRowKey(row, rowIndex) : rowIndex}>
              {columns.map((column, colIndex) => (
                <TableCell key={colIndex}>
                  {column.cell
                    ? column.cell(row)
                    : String(row[column.accessor as keyof T] ?? "")}
                </TableCell>
              ))}
            </TableRow>
          ))
        ) : (
          <TableRow>
            <TableCell colSpan={columns.length} className="text-center">
              <span className="text-muted-foreground">{emptyMessage}</span>
            </TableCell>
          </TableRow>
        )}
      </TableBody>
    </Table>
  );
}
