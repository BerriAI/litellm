import { ColumnDef, flexRender, getCoreRowModel, SortingState, useReactTable } from "@tanstack/react-table";
import React from "react";
import { Table, TableHead, TableHeaderCell, TableBody, TableRow, TableCell, Select, SelectItem } from "@tremor/react";
import { SwitchVerticalIcon, ChevronUpIcon, ChevronDownIcon } from "@heroicons/react/outline";
import { Skeleton } from "antd";
import { UserInfo } from "./types";
import UserInfoView from "./user_info_view";
import { columns as createColumns } from "./columns";
import { FilterInput } from "@/components/common_components/Filters/FilterInput";
import { FiltersButton } from "@/components/common_components/Filters/FiltersButton";
import { ResetFiltersButton } from "@/components/common_components/Filters/ResetFiltersButton";
import { Search, User, CircleUserRound } from "lucide-react";

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
    onSortingChange: (updaterOrValue: any) => {
      const newSorting = typeof updaterOrValue === "function" ? updaterOrValue(sorting) : updaterOrValue;
      setSorting(newSorting);
      if (newSorting && Array.isArray(newSorting) && newSorting.length > 0 && newSorting[0]) {
        const sortState = newSorting[0];
        if (sortState.id) {
          const sortBy = sortState.id;
          const sortOrder = sortState.desc ? "desc" : "asc";
          onSortChange?.(sortBy, sortOrder);
        }
      } else {
        // Reset to default sort when no sorting is selected
        onSortChange?.("created_at", "desc");
      }
    },
    getCoreRowModel: getCoreRowModel(),
    manualSorting: true,
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
            <FilterInput
              placeholder="Search by email..."
              value={filters.email}
              onChange={(value) => updateFilters({ email: value })}
              icon={Search}
            />

            {/* Filter Button */}
            <FiltersButton
              onClick={() => setShowFilters(!showFilters)}
              active={showFilters}
              hasActiveFilters={!!(filters.user_id || filters.user_role || filters.team)}
            />

            {/* Reset Filters Button */}
            <ResetFiltersButton
              onClick={() => {
                updateFilters(initialFilters);
              }}
            />
          </div>

          {/* Additional Filters */}
          {showFilters && (
            <div className="flex flex-wrap items-center gap-3 mt-3">
              {/* User ID Search */}
              <FilterInput
                placeholder="Filter by User ID"
                value={filters.user_id}
                onChange={(value) => updateFilters({ user_id: value })}
                icon={User}
              />

              <FilterInput
                placeholder="Filter by SSO ID"
                value={filters.sso_user_id}
                onChange={(value) => updateFilters({ sso_user_id: value })}
                icon={CircleUserRound}
              />

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
            </div>
          )}

          {/* Results Count and Pagination */}
          <div className="flex justify-between items-center">
            {isLoading ? (
              <Skeleton.Input active style={{ width: 192, height: 20 }} />
            ) : (
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
            )}

            {/* Pagination Buttons */}
            <div className="flex space-x-2">
              {isLoading ? (
                <>
                  <Skeleton.Button active size="small" style={{ width: 80, height: 30 }} />
                  <Skeleton.Button active size="small" style={{ width: 60, height: 30 }} />
                </>
              ) : (
                <>
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
                </>
              )}
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
                        } ${header.column.getCanSort() ? "cursor-pointer hover:bg-gray-50" : ""}`}
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center">
                            {header.isPlaceholder
                              ? null
                              : flexRender(header.column.columnDef.header, header.getContext())}
                          </div>
                          {header.id !== "actions" && header.column.getCanSort() && (
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
