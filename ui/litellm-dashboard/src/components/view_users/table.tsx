import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import React from "react";
import { Table, TableHead, TableHeaderCell, TableBody, TableRow, TableCell, Select, SelectItem } from "@tremor/react";
import { SwitchVerticalIcon, ChevronUpIcon, ChevronDownIcon } from "@heroicons/react/outline";
import { UserInfo } from "./types";
import UserInfoView from "./user_info_view";
import { columns as createColumns } from "./columns";

interface FilterState {
  email: string;
  user_id: string;
  user_role: string;
  sso_user_id: string;
  team: string;
  model: string;
  min_spend: number | null;
  max_spend: number | null;
  sort_by: string;
  sort_order: "asc" | "desc";
}

interface UserDataTableProps {
  data: UserInfo[];
  columns: ColumnDef<UserInfo, any>[];
  isLoading?: boolean;
  onSortChange?: (sortBy: string, sortOrder: "asc" | "desc") => void;
  currentSort?: {
    sortBy: string;
    sortOrder: "asc" | "desc";
  };
  accessToken: string | null;
  userRole: string | null;
  possibleUIRoles: Record<string, Record<string, string>> | null;
  handleEdit: (user: UserInfo) => void;
  handleDelete: (user: UserInfo) => void;
  handleResetPassword: (userId: string) => void;
  selectedUsers?: UserInfo[];
  onSelectionChange?: (selectedUsers: UserInfo[]) => void;
  enableSelection?: boolean;
  // Filter-related props
  filters: FilterState;
  updateFilters: (update: Partial<FilterState>) => void;
  initialFilters: FilterState;
  teams: any[] | null;
  // Pagination props
  userListResponse: any;
  currentPage: number;
  handlePageChange: (newPage: number) => void;
}

export function UserDataTable({
  data = [],
  columns: originalColumns,
  isLoading = false,
  onSortChange,
  currentSort,
  accessToken,
  userRole,
  possibleUIRoles,
  handleEdit,
  handleDelete,
  handleResetPassword,
  selectedUsers = [],
  onSelectionChange,
  enableSelection = false,
  filters,
  updateFilters,
  initialFilters,
  teams,
  userListResponse,
  currentPage,
  handlePageChange,
}: UserDataTableProps) {
  const [sorting, setSorting] = React.useState<SortingState>([
    {
      id: currentSort?.sortBy || "created_at",
      desc: currentSort?.sortOrder === "desc",
    },
  ]);
  const [selectedUserId, setSelectedUserId] = React.useState<string | null>(null);
  const [openInEditMode, setOpenInEditMode] = React.useState<boolean>(false);
  const [showFilters, setShowFilters] = React.useState<boolean>(false);

  const handleUserClick = (userId: string, openInEditMode: boolean = false) => {
    setSelectedUserId(userId);
    setOpenInEditMode(openInEditMode);
  };

  const handleCloseUserInfo = () => {
    setSelectedUserId(null);
    setOpenInEditMode(false);
  };

  // Selection handlers
  const handleSelectUser = (user: UserInfo, isSelected: boolean) => {
    if (!onSelectionChange) return;

    if (isSelected) {
      onSelectionChange([...selectedUsers, user]);
    } else {
      onSelectionChange(selectedUsers.filter((u) => u.user_id !== user.user_id));
    }
  };

  const handleSelectAll = (isSelected: boolean) => {
    if (!onSelectionChange) return;

    if (isSelected) {
      onSelectionChange(data);
    } else {
      onSelectionChange([]);
    }
  };

  const isUserSelected = (user: UserInfo) => {
    return selectedUsers.some((u) => u.user_id === user.user_id);
  };

  const isAllSelected = data.length > 0 && selectedUsers.length === data.length;
  const isIndeterminate = selectedUsers.length > 0 && selectedUsers.length < data.length;

  // Create columns with the handleUserClick function
  const columns = React.useMemo(() => {
    if (possibleUIRoles) {
      return createColumns(
        possibleUIRoles,
        handleEdit,
        handleDelete,
        handleResetPassword,
        handleUserClick,
        enableSelection
          ? {
              selectedUsers,
              onSelectUser: handleSelectUser,
              onSelectAll: handleSelectAll,
              isUserSelected,
              isAllSelected,
              isIndeterminate,
            }
          : undefined,
      );
    }
    return originalColumns;
  }, [
    possibleUIRoles,
    handleEdit,
    handleDelete,
    handleResetPassword,
    handleUserClick,
    originalColumns,
    enableSelection,
    selectedUsers,
    isAllSelected,
    isIndeterminate,
  ]);

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
        const sortOrder = sortState.desc ? "desc" : "asc";
        onSortChange?.(sortBy, sortOrder);
      }
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableSorting: true,
  });

  // Update local sorting state when currentSort prop changes
  React.useEffect(() => {
    if (currentSort) {
      setSorting([
        {
          id: currentSort.sortBy,
          desc: currentSort.sortOrder === "desc",
        },
      ]);
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
        initialTab={openInEditMode ? 1 : 0}
        startInEditMode={openInEditMode}
      />
    );
  }

  return (
    <div className="bg-white rounded-lg shadow">
      {/* Filter Section */}
      <div className="border-b px-6 py-4">
        <div className="flex flex-col space-y-4">
          {/* Search and Filter Controls */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Email Search */}
            <div className="relative w-64">
              <input
                type="text"
                placeholder="Search by email..."
                className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                value={filters.email}
                onChange={(e) => updateFilters({ email: e.target.value })}
              />
              <svg
                className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
            </div>

            {/* Filter Button */}
            <button
              className={`px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 ${showFilters ? "bg-gray-100" : ""}`}
              onClick={() => setShowFilters(!showFilters)}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
                />
              </svg>
              Filters
              {(filters.user_id || filters.user_role || filters.team) && (
                <span className="w-2 h-2 rounded-full bg-blue-500"></span>
              )}
            </button>

            {/* Reset Filters Button */}
            <button
              className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
              onClick={() => {
                updateFilters(initialFilters);
              }}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              Reset Filters
            </button>
          </div>

          {/* Additional Filters */}
          {showFilters && (
            <div className="flex flex-wrap items-center gap-3 mt-3">
              {/* User ID Search */}
              <div className="relative w-64">
                <input
                  type="text"
                  placeholder="Filter by User ID"
                  className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  value={filters.user_id}
                  onChange={(e) => updateFilters({ user_id: e.target.value })}
                />
                <svg
                  className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5.121 17.804A13.937 13.937 0 0112 16c2.5 0 4.847.655 6.879 1.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>

              {/* Role Dropdown */}
              <div className="w-64">
                <Select
                  value={filters.user_role}
                  onValueChange={(value) => updateFilters({ user_role: value })}
                  placeholder="Select Role"
                >
                  {possibleUIRoles &&
                    Object.entries(possibleUIRoles).map(([key, value]) => (
                      <SelectItem key={key} value={key}>
                        {value.ui_label}
                      </SelectItem>
                    ))}
                </Select>
              </div>

              {/* Team Dropdown */}
              <div className="w-64">
                <Select
                  value={filters.team}
                  onValueChange={(value) => updateFilters({ team: value })}
                  placeholder="Select Team"
                >
                  {teams?.map((team) => (
                    <SelectItem key={team.team_id} value={team.team_id}>
                      {team.team_alias || team.team_id}
                    </SelectItem>
                  ))}
                </Select>
              </div>

              {/* SSO ID Search */}
              <div className="relative w-64">
                <input
                  type="text"
                  placeholder="Filter by SSO ID"
                  className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  value={filters.sso_user_id}
                  onChange={(e) => updateFilters({ sso_user_id: e.target.value })}
                />
              </div>
            </div>
          )}

          {/* Results Count and Pagination */}
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-700">
              Showing{" "}
              {userListResponse && userListResponse.users && userListResponse.users.length > 0
                ? (userListResponse.page - 1) * userListResponse.page_size + 1
                : 0}{" "}
              -{" "}
              {userListResponse && userListResponse.users
                ? Math.min(userListResponse.page * userListResponse.page_size, userListResponse.total)
                : 0}{" "}
              of {userListResponse ? userListResponse.total : 0} results
            </span>

            {/* Pagination Buttons */}
            <div className="flex space-x-2">
              <button
                onClick={() => handlePageChange(currentPage - 1)}
                disabled={currentPage === 1}
                className={`px-3 py-1 text-sm border rounded-md ${
                  currentPage === 1 ? "bg-gray-100 text-gray-400 cursor-not-allowed" : "hover:bg-gray-50"
                }`}
              >
                Previous
              </button>
              <button
                onClick={() => handlePageChange(currentPage + 1)}
                disabled={!userListResponse || currentPage >= userListResponse.total_pages}
                className={`px-3 py-1 text-sm border rounded-md ${
                  !userListResponse || currentPage >= userListResponse.total_pages
                    ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                    : "hover:bg-gray-50"
                }`}
              >
                Next
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Table Section */}
      <div className="overflow-auto">
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
                          header.id === "actions"
                            ? "sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                            : ""
                        }`}
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center">
                            {header.isPlaceholder
                              ? null
                              : flexRender(header.column.columnDef.header, header.getContext())}
                          </div>
                          {header.id !== "actions" && (
                            <div className="w-4">
                              {header.column.getIsSorted() ? (
                                {
                                  asc: <ChevronUpIcon className="h-4 w-4 text-blue-500" />,
                                  desc: <ChevronDownIcon className="h-4 w-4 text-blue-500" />,
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
                            cell.column.id === "actions"
                              ? "sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                              : ""
                          }`}
                          onClick={() => {
                            if (cell.column.id === "user_id") {
                              handleUserClick(cell.getValue() as string, false);
                            }
                          }}
                          style={{
                            cursor: cell.column.id === "user_id" ? "pointer" : "default",
                            color: cell.column.id === "user_id" ? "#3b82f6" : "inherit",
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
      </div>
    </div>
  );
}
