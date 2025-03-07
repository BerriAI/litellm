import { Fragment } from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getExpandedRowModel,
  Row,
  useReactTable,
} from "@tanstack/react-table";

import {
  Table,
  TableHead,
  TableHeaderCell,
  TableBody,
  TableRow,
  TableCell,
} from "@tremor/react";

interface DataTableProps<TData, TValue> {
  data: TData[];
  columns: ColumnDef<TData, TValue>[];
  renderSubComponent: (props: { row: Row<TData> }) => React.ReactElement;
  getRowCanExpand: (row: Row<TData>) => boolean;
  isLoading?: boolean;
}

export function DataTable<TData, TValue>({
  data = [],
  columns,
  getRowCanExpand,
  renderSubComponent,
  isLoading = false,
}: DataTableProps<TData, TValue>) {
  const table = useReactTable({
    data,
    columns,
    getRowCanExpand,
    getCoreRowModel: getCoreRowModel(),
    getExpandedRowModel: getExpandedRowModel(),
  });

  return (
    <div className="rounded-lg custom-border">
      <Table className="[&_td]:py-0.5 [&_th]:py-1">
        <TableHead>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id}>
              {headerGroup.headers.map((header) => {
                return (
                  <TableHeaderCell key={header.id} className="py-1 h-8">
                    {header.isPlaceholder ? null : (
                      flexRender(
                        header.column.columnDef.header,
                        header.getContext()
                      )
                    )}
                  </TableHeaderCell>
                );
              })}
            </TableRow>
          ))}
        </TableHead>
        <TableBody>
          {isLoading ?
            <TableRow>
              <TableCell colSpan={columns.length} className="h-8 text-center">
                <div className="text-center text-gray-500">
                  <p>🚅 Loading logs...</p>
                </div>
              </TableCell>
            </TableRow>
          : table.getRowModel().rows.length > 0 ?
            table.getRowModel().rows.map((row) => (
              <Fragment key={row.id}>
                <TableRow className="h-8">
                  {row.getVisibleCells().map((cell) => (
                    <TableCell 
                      key={cell.id} 
                      className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap"
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </TableCell>
                  ))}
                </TableRow>

                {row.getIsExpanded() && (
                  <TableRow>
                    <TableCell colSpan={row.getVisibleCells().length}>
                      {renderSubComponent({ row })}
                    </TableCell>
                  </TableRow>
                )}
              </Fragment>
            ))
          : <TableRow>
              <TableCell colSpan={columns.length} className="h-8 text-center">
                <div className="text-center text-gray-500">
                  <p>No logs found</p>
                </div>
              </TableCell>
            </TableRow>
          }
        </TableBody>
      </Table>
    </div>
  );
}
