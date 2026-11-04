"use client";

import {
  ColumnFiltersState,
  OnChangeFn,
  PaginationState,
  RowSelectionState,
  SortingState,
} from "@tanstack/react-table";
import type { ParseKeys } from "i18next";
import { Users } from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

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
  canViewUserUsage: boolean;
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

const FILTER_LABEL_KEYS: Readonly<Record<string, ParseKeys>> = {
  user_id: "users.table.filters.userId",
  sso_user_id: "users.table.filters.ssoId",
  user_role: "users.table.filters.role",
  team: "users.table.filters.team",
};

function EmptyState() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Users className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{t("users.table.emptyState.title")}</div>
      <div className="text-sm text-muted-foreground">{t("users.table.emptyState.description")}</div>
    </div>
  );
}

export function UsersTable({
  data,
  rowCount,
  isLoading,
  possibleUIRoles,
  teams,
  canViewUserUsage,
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
  const { t } = useTranslation();
  const [filtersOpen, setFiltersOpen] = useState(false);

  const columns = useMemo(() => {
    const columnDeps = {
      possibleUIRoles,
      includeSelection: selectionEnabled,
      canViewUserUsage,
      onUserClick,
      onDeleteUser,
      onResetPassword,
      t,
    };
    return getUsersTableColumns(columnDeps);
  }, [possibleUIRoles, selectionEnabled, canViewUserUsage, onUserClick, onDeleteUser, onResetPassword, t]);

  const filterLabels = useMemo(
    () => Object.fromEntries(Object.entries(FILTER_LABEL_KEYS).map(([columnId, key]) => [columnId, t(key)])),
    [t],
  );

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
      loadingMessage={t("users.table.loading")}
      noDataMessage={<EmptyState />}
      size="compact"
      toolbar={(table) => (
        <>
          <DataTableToolbar
            table={table}
            searchValue={searchValue}
            onSearchChange={onSearchChange}
            searchPlaceholder={t("users.table.searchPlaceholder")}
            onOpenFilters={() => setFiltersOpen(true)}
            filterLabels={filterLabels}
            formatFilterValue={formatFilterValue}
          />
          <DataTableFilterDrawer
            table={table}
            open={filtersOpen}
            onOpenChange={setFiltersOpen}
            title={t("users.table.filters.title")}
            description={t("users.table.filters.description")}
          >
            {({ get, set }) => (
              <>
                <DataTableFilterField label={t("users.table.filters.userId")}>
                  <Input
                    value={(get("user_id") as string) ?? ""}
                    onChange={(event) => set("user_id", event.target.value)}
                    placeholder={t("users.table.filters.userIdPlaceholder")}
                    data-testid="users-filter-user-id"
                  />
                </DataTableFilterField>
                <DataTableFilterField label={t("users.table.filters.ssoId")}>
                  <Input
                    value={(get("sso_user_id") as string) ?? ""}
                    onChange={(event) => set("sso_user_id", event.target.value)}
                    placeholder={t("users.table.filters.ssoIdPlaceholder")}
                    data-testid="users-filter-sso-id"
                  />
                </DataTableFilterField>
                <DataTableFilterField label={t("users.table.filters.role")}>
                  <SearchSelect
                    options={roleOptions}
                    value={(get("user_role") as string) || undefined}
                    onValueChange={(value) => set("user_role", value ?? undefined)}
                    placeholder={t("users.table.filters.rolePlaceholder")}
                    emptyText={t("users.table.filters.roleEmpty")}
                  />
                </DataTableFilterField>
                <DataTableFilterField label={t("users.table.filters.team")}>
                  <SearchSelect
                    options={teamOptions}
                    value={(get("team") as string) || undefined}
                    onValueChange={(value) => set("team", value ?? undefined)}
                    placeholder={t("users.table.filters.teamPlaceholder")}
                    emptyText={t("users.table.filters.teamEmpty")}
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
