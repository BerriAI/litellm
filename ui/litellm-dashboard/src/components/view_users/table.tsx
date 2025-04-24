import { Fragment } from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
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
import { SwitchVerticalIcon, ChevronUpIcon, ChevronDownIcon } from "@heroicons/react/outline";
import { UserInfo } from "./types";
import UserInfoView from "./user_info_view";

interface UserDataTableProps {
  data: UserInfo[];
  columns: ColumnDef<UserInfo, any>[];
  isLoading?: boolean;
  onSortChange?: (sortBy: string, sortOrder: 'asc' | 'desc') => void;
  currentSort?: {
    sortBy: string;
    sortOrder: 'asc' | 'desc';
  };
  accessToken: string | null;
  userRole: string | null;
  possibleUIRoles: Record<string, Record<string, string>> | null;
}

export function UserDataTable({
  data = [],
  columns,
  isLoading = false,
  onSortChange,
  currentSort,
  accessToken,
  userRole,
  possibleUIRoles,
}: UserDataTableProps) {
  const [sorting, setSorting] = React.useState<SortingState>([
    { 
      id: currentSort?.sortBy || "created_at", 
      desc: currentSort?.sortOrder === "desc" 
    }
  ]);
  const [selectedUserId, setSelectedUserId] = React.useState<string | null>(null);

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting,
    },
    onSortingChange: (newSorting: any) => {
      setSorting(newSorting);
      if (newSorting.length > 0) {
        const sortState = newSorting[0];
        const sortBy = sortState.id;
        const sortOrder = sortState.desc ? 'desc' : 'asc';
        onSortChange?.(sortBy, sortOrder);
      }
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableSorting: true,
  });

  const handleUserClick = (userId: string) => {
    setSelectedUserId(userId);
  };

  const handleCloseUserInfo = () => {
    setSelectedUserId(null);
  };

  // Update local sorting state when currentSort prop changes
  React.useEffect(() => {
    if (currentSort) {
      setSorting([{
        id: currentSort.sortBy,
        desc: currentSort.sortOrder === 'desc'
      }]);
    }
  }, [currentSort]);

  if (selectedUserId) {
    return (
      <UserInfoView
        userId={selectedUserId}
        onClose={handleCloseUserInfo}
        accessToken={accessToken}
        userRole={userRole}
        possibleUIRoles={possibleUIRoles}
      />
    );
  }

  return (
    <div className="rounded-lg custom-border relative">
      <div className="overflow-x-auto">
        <Table className="[&_td]:py-0.5 [&_th]:py-1">
          <TableHead>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHeaderCell 
                    key={header.id} 
                    className={`py-1 h-8 ${
                      header.id === 'actions' 
                        ? 'sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]' 
                        : ''
                    }`}
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
                    <p>🚅 Loading users...</p>
                  </div>
                </TableCell>
              </TableRow>
            ) : data.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} className="h-8">
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      className={`py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap ${
                        cell.column.id === 'actions'
                          ? 'sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]'
                          : ''
                      }`}
                      onClick={() => {
                        if (cell.column.id === 'user_id') {
                          handleUserClick(cell.getValue() as string);
                        }
                      }}
                      style={{
                        cursor: cell.column.id === 'user_id' ? 'pointer' : 'default',
                        color: cell.column.id === 'user_id' ? '#3b82f6' : 'inherit',
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
                    <p>No users found</p>
                  </div>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
} 