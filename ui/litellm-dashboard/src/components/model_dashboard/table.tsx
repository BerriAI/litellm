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
}

export function ModelDataTable<TData, TValue>({
  data = [],
  columns,
  isLoading = false,
}: ModelDataTableProps<TData, TValue>) {
  const [sorting, setSorting] = React.useState<SortingState>([
    { id: "model_info.created_at", desc: true }
  ]);
  const [columnResizeMode] = React.useState<ColumnResizeMode>("onChange");
  const [columnSizing, setColumnSizing] = React.useState({});
  const [columnVisibility, setColumnVisibility] = React.useState<VisibilityState>({});
  const [isDropdownOpen, setIsDropdownOpen] = React.useState(false);
  const dropdownRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const table = useReactTable({
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
    <div className="space-y-4">
      <div className="flex justify-end">
        <div className="relative" ref={dropdownRef}>
          <button
            onClick={() => setIsDropdownOpen(!isDropdownOpen)}
            className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <TableIcon className="h-4 w-4" />
            Columns
          </button>
          {isDropdownOpen && (
            <div className="absolute right-0 mt-2 w-56 bg-white rounded-md shadow-lg ring-1 ring-black ring-opacity-5 z-50">
              <div className="py-1">
                {table.getAllLeafColumns().map((column) => {
                  if (column.id === 'actions') return null;
                  return (
                    <div
                      key={column.id}
                      className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                      onClick={() => column.toggleVisibility()}
                    >
                      <input
                        type="checkbox"
                        checked={column.getIsVisible()}
                        onChange={() => column.toggleVisibility()}
                        className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="ml-2">{getHeaderText(column.columnDef.header)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="rounded-lg custom-border relative">
        <div className="overflow-x-auto">
          <div className="relative min-w-full">
            <Table className="[&_td]:py-0.5 [&_th]:py-1 w-full">
              <TableHead>
                {table.getHeaderGroups().map((headerGroup) => (
                  <TableRow key={headerGroup.id}>
                    {headerGroup.headers.map((header) => (
                      <TableHeaderCell 
                        key={header.id} 
                        className={`py-1 h-8 relative ${
                          header.id === 'actions' 
                            ? 'sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] z-10 w-[120px] ml-8' 
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
                ) : table.getRowModel().rows.length > 0 ? (
                  table.getRowModel().rows.map((row) => (
                    <TableRow key={row.id} className="h-8">
                      {row.getVisibleCells().map((cell) => (
                        <TableCell
                          key={cell.id}
                          className={`py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap ${
                            cell.column.id === 'actions'
                              ? 'sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] z-10 w-[120px] ml-8'
                              : ''
                          }`}
                          style={{
                            width: cell.column.id === 'actions' ? 120 : cell.column.getSize(),
                            minWidth: cell.column.id === 'actions' ? 120 : cell.column.getSize(),
                            maxWidth: cell.column.id === 'actions' ? 120 : cell.column.getSize(),
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
    </div>
  );
} 