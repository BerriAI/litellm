import { Fragment } from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getExpandedRowModel,
  Row,
  useReactTable,
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
import { TableIcon } from "@heroicons/react/outline";

interface DataTableProps<TData, TValue> {
  data: TData[];
  columns: ColumnDef<TData, TValue>[];
  renderSubComponent: (props: { row: Row<TData> }) => React.ReactElement;
  getRowCanExpand: (row: Row<TData>) => boolean;
  isLoading?: boolean;
  expandedRequestId?: string | null;
  onRowExpand?: (requestId: string | null) => void;
  setSelectedKeyIdInfoView?: (keyId: string | null) => void;
}

export function DataTable<TData extends { request_id: string }, TValue>({
  data = [],
  columns,
  getRowCanExpand,
  renderSubComponent,
  isLoading = false,
  expandedRequestId,
  onRowExpand,
}: DataTableProps<TData, TValue>) {
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
    getRowCanExpand,
    getCoreRowModel: getCoreRowModel(),
    getExpandedRowModel: getExpandedRowModel(),
    state: {
      expanded: expandedRequestId 
        ? data.reduce((acc, row, index) => {
            if (row.request_id === expandedRequestId) {
              acc[index] = true;
            }
            return acc;
          }, {} as Record<string, boolean>)
        : {},
      columnVisibility,
    },
    onExpandedChange: (updater) => {
      if (!onRowExpand) return;
      
      const currentExpanded = expandedRequestId 
        ? data.reduce((acc, row, index) => {
            if (row.request_id === expandedRequestId) {
              acc[index] = true;
            }
            return acc;
          }, {} as Record<string, boolean>)
        : {};
      
      const newExpanded = typeof updater === 'function' 
        ? updater(currentExpanded) 
        : updater;
      
      if (Object.keys(newExpanded).length === 0) {
        onRowExpand(null);
        return;
      }
      
      const expandedIndex = Object.keys(newExpanded)[0];
      const expandedRow = expandedIndex !== undefined ? data[parseInt(expandedIndex)] : null;
      
      onRowExpand(expandedRow ? expandedRow.request_id : null);
    },
    onColumnVisibilityChange: setColumnVisibility,
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
    <div className="relative w-full">
      {/* Column visibility toggle */}
      <div className="my-2" ref={dropdownRef}>
        <button
          onClick={() => setIsDropdownOpen(!isDropdownOpen)}
          className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          <TableIcon className="h-4 w-4" />
          Columns
        </button>
        {isDropdownOpen && (
          <div className="absolute left-0 mt-2 w-56 bg-white rounded-md shadow-lg ring-1 ring-black ring-opacity-5 z-50">
            <div className="py-1 max-h-[60vh] overflow-y-auto">
              {table.getAllLeafColumns().map((column) => {
                if (column.id === 'expander') return null;
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

      <div className="rounded-lg custom-border clear-both">
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
    </div>
  );
}
