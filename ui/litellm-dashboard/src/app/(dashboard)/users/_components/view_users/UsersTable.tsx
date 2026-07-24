"use client";

import {
  ColumnFiltersState,
  OnChangeFn,
  PaginationState,
  RowSelectionState,
  SortingState,
} from "@tanstack/react-table";
import { Users } from "lucide-react";
import { useMemo, useState } from "react";

import { UserInfo } from "@/components/networking";
import {
  DataTable,
  DataTableFilterDrawer,
  DataTableFilterField,
  DataTableToolbar,
} from "@/components/shared/DataTable";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Input } from "@/components/ui/input";

import { getUsersTableColumns } from "./UsersTableColumns";

export interface UsersTableTeamOption {
  team_id: string;
  team_alias?: string | null;
}

interface UsersTableProps {
  data: UserInfo[];
  rowCount: number;
  isLoading: boolean;
  possibleUIRoles: Record<string, Record<string, string>> | null;
  teams: UsersTableTeamOption[] | null;
  sorting: SortingState;
  onSortingChange: OnChangeFn<SortingState>;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
  columnFilters: ColumnFiltersState;
  onColumnFiltersChange: OnChangeFn<ColumnFiltersState>;
  searchValue: string;
  onSearchChange: (value: string) => void;
  selectionEnabled: boolean;
  rowSelection: RowSelectionState;
  onRowSelectionChange: OnChangeFn<RowSelectionState>;
  onUserClick: (userId: string, openInEditMode?: boolean) => void;
  onDeleteUser: (user: UserInfo) => void;
  onResetPassword: (userId: string) => void;
}

const FILTER_LABELS: Record<string, string> = {
  user_id: "User ID",
  sso_user_id: "SSO ID",
  user_role: "Role",
  team: "Team",
};

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Users className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No users found</div>
      <div className="text-sm text-muted-foreground">Try adjusting your search or filters.</div>
    </div>
  );
}

export function UsersTable({
  data,
  rowCount,
  isLoading,
  possibleUIRoles,
  teams,
  sorting,
  onSortingChange,
  pagination,
  onPaginationChange,
  columnFilters,
  onColumnFiltersChange,
  searchValue,
  onSearchChange,
  selectionEnabled,
  rowSelection,
  onRowSelectionChange,
  onUserClick,
  onDeleteUser,
  onResetPassword,
}: UsersTableProps) {
  const [filtersOpen, setFiltersOpen] = useState(false);

  const columns = useMemo(() => {
    const columnDeps = {
      possibleUIRoles,
      includeSelection: selectionEnabled,
      onUserClick,
      onDeleteUser,
      onResetPassword,
    };
    return getUsersTableColumns(columnDeps);
  }, [possibleUIRoles, selectionEnabled, onUserClick, onDeleteUser, onResetPassword]);

  const roleOptions = useMemo(
    () =>
      Object.entries(possibleUIRoles ?? {}).map(([role, config]) => ({
        label: config.ui_label || role,
        value: role,
      })),
    [possibleUIRoles],
  );

  const teamOptions = useMemo(
    () =>
      (teams ?? []).map((team) => ({
        label: team.team_alias || team.team_id,
        value: team.team_id,
      })),
    [teams],
  );

  const formatFilterValue = (columnId: string, value: unknown): string => {
    const raw = String(value);
    if (columnId === "user_role") {
      return possibleUIRoles?.[raw]?.ui_label || raw;
    }
    if (columnId === "team") {
      return teams?.find((team) => team.team_id === raw)?.team_alias || raw;
    }
    return raw;
  };

  return (
    <DataTable
      data={data}
      columns={columns}
      getRowId={(row) => row.user_id}
      sortingMode="server"
      sorting={sorting}
      onSortingChange={onSortingChange}
      paginationMode="server"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      rowCount={rowCount}
      filterMode="server"
      columnFilters={columnFilters}
      onColumnFiltersChange={onColumnFiltersChange}
      rowSelection={rowSelection}
      onRowSelectionChange={onRowSelectionChange}
      isLoading={isLoading}
      loadingMessage="Loading users…"
      noDataMessage={<EmptyState />}
      size="compact"
      toolbar={(table) => (
        <>
          <DataTableToolbar
            table={table}
            searchValue={searchValue}
            onSearchChange={onSearchChange}
            searchPlaceholder="Search by email…"
            onOpenFilters={() => setFiltersOpen(true)}
            filterLabels={FILTER_LABELS}
            formatFilterValue={formatFilterValue}
          />
          <DataTableFilterDrawer
            table={table}
            open={filtersOpen}
            onOpenChange={setFiltersOpen}
            title="Filters"
            description="Narrow down your users"
          >
            {({ get, set }) => (
              <>
                <DataTableFilterField label="User ID">
                  <Input
                    value={(get("user_id") as string) ?? ""}
                    onChange={(event) => set("user_id", event.target.value)}
                    placeholder="Enter user ID…"
                    data-testid="users-filter-user-id"
                  />
                </DataTableFilterField>
                <DataTableFilterField label="SSO ID">
                  <Input
                    value={(get("sso_user_id") as string) ?? ""}
                    onChange={(event) => set("sso_user_id", event.target.value)}
                    placeholder="Enter SSO ID…"
                    data-testid="users-filter-sso-id"
                  />
                </DataTableFilterField>
                <DataTableFilterField label="Role">
                  <SearchSelect
                    options={roleOptions}
                    value={(get("user_role") as string) || undefined}
                    onValueChange={(value) => set("user_role", value)}
                    placeholder="Select a role…"
                    emptyText="No roles found"
                  />
                </DataTableFilterField>
                <DataTableFilterField label="Team">
                  <SearchSelect
                    options={teamOptions}
                    value={(get("team") as string) || undefined}
                    onValueChange={(value) => set("team", value)}
                    placeholder="Select a team…"
                    emptyText="No teams found"
                  />
                </DataTableFilterField>
              </>
            )}
          </DataTableFilterDrawer>
        </>
      )}
    />
  );
}
