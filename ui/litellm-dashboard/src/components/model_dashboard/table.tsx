import { Fragment } from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
  ColumnResizeMode,
  VisibilityState,
} from "@tanstack/react-table";
import React from "react";
import {
  Table,
  TableHead,
  TableHeaderCell,
  TableBody,
  TableRow,
  TableCell,
} from "@tremor/react";
import { SwitchVerticalIcon, ChevronUpIcon, ChevronDownIcon, TableIcon } from "@heroicons/react/outline";

interface ModelDataTableProps<TData, TValue> {
  data: TData[];
  columns: ColumnDef<TData, TValue>[];
  isLoading?: boolean;
  table: any; // Add table prop to access column visibility controls
}

export function ModelDataTable<TData, TValue>({
  data = [],
  columns,
  isLoading = false,
  table
}: ModelDataTableProps<TData, TValue>) {
  const [sorting, setSorting] = React.useState<SortingState>([
    { id: "model_info.created_at", desc: true }
  ]);
  const [columnResizeMode] = React.useState<ColumnResizeMode>("onChange");
  const [columnSizing, setColumnSizing] = React.useState({});
  const [columnVisibility, setColumnVisibility] = React.useState<VisibilityState>({});

  const tableInstance = useReactTable({
    data,
    columns,
    state: {
      sorting,
      columnSizing,
      columnVisibility,
    },
    columnResizeMode,
    onSortingChange: setSorting,
    onColumnSizingChange: setColumnSizing,
    onColumnVisibilityChange: setColumnVisibility,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableSorting: true,
    enableColumnResizing: true,
    defaultColumn: {
      minSize: 40,
      maxSize: 500,
    },
  });

  // Expose table instance to parent
  React.useEffect(() => {
    if (table) {
      table.current = tableInstance;
    }
  }, [tableInstance, table]);

  const getHeaderText = (header: any): string => {
    if (typeof header === 'string') {
      return header;
    }
    if (typeof header === 'function') {
      const headerElement = header();
      if (headerElement && headerElement.props && headerElement.props.children) {
        const children = headerElement.props.children;
        if (typeof children === 'string') {
          return children;
        }
        if (children.props && children.props.children) {
          return children.props.children;
        }
      }
    }
    return '';
  };

  return (
    <div className="rounded-lg custom-border relative">
      <div className="overflow-x-auto">
        <div className="relative min-w-full">
          <Table className="[&_td]:py-0.5 [&_th]:py-1 w-full">
            <TableHead>
              {tableInstance.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header) => (
                    <TableHeaderCell 
                      key={header.id} 
                      className={`py-1 h-8 relative ${
                        header.id === 'actions' 
                          ? 'sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] z-20 w-[120px] ml-8' 
                          : ''
                      }`}
                      style={{
                        width: header.id === 'actions' ? 120 : header.getSize(),
                        position: header.id === 'actions' ? 'sticky' : 'relative',
                        right: header.id === 'actions' ? 0 : 'auto',
                      }}
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center">
                          {header.isPlaceholder ? null : (
                            flexRender(
                              header.column.columnDef.header,
                              header.getContext()
                            )
                          )}
                        </div>
                        {header.id !== 'actions' && (
                          <div className="w-4">
                            {header.column.getIsSorted() ? (
                              {
                                asc: <ChevronUpIcon className="h-4 w-4 text-blue-500" />,
                                desc: <ChevronDownIcon className="h-4 w-4 text-blue-500" />
                              }[header.column.getIsSorted() as string]
                            ) : (
                              <SwitchVerticalIcon className="h-4 w-4 text-gray-400" />
                            )}
                          </div>
                        )}
                      </div>
                      {header.column.getCanResize() && (
                        <div
                          onMouseDown={header.getResizeHandler()}
                          onTouchStart={header.getResizeHandler()}
                          className={`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${
                            header.column.getIsResizing() ? 'bg-blue-500' : 'hover:bg-blue-200'
                          }`}
                        />
                      )}
                    </TableHeaderCell>
                  ))}
                </TableRow>
              ))}
            </TableHead>
            <TableBody>
              {isLoading ? (
                <TableRow>
                  <TableCell colSpan={columns.length} className="h-8 text-center">
                    <div className="text-center text-gray-500">
                      <p>🚅 Loading models...</p>
                    </div>
                  </TableCell>
                </TableRow>
              ) : tableInstance.getRowModel().rows.length > 0 ? (
                tableInstance.getRowModel().rows.map((row) => (
                  <TableRow key={row.id}>
                    {row.getVisibleCells().map((cell) => (
                      <TableCell
                        key={cell.id}
                        className={`py-0.5 ${
                          cell.column.id === 'actions'
                            ? 'sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] z-20 w-[120px] ml-8'
                            : ''
                        }`}
                        style={{
                          width: cell.column.id === 'actions' ? 120 : cell.column.getSize(),
                          position: cell.column.id === 'actions' ? 'sticky' : 'relative',
                          right: cell.column.id === 'actions' ? 0 : 'auto',
                        }}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    ))}
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={columns.length} className="h-8 text-center">
                    <div className="text-center text-gray-500">
                      <p>No models found</p>
                    </div>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  );
} 