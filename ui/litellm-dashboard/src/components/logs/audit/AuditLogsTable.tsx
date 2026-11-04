"use client";

import { ColumnFiltersState, OnChangeFn, PaginationState } from "@tanstack/react-table";
import type { TFunction } from "i18next";
import { ScrollText } from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  DataTable,
  DataTableFilterDrawer,
  DataTableFilterField,
  DataTableToolbar,
} from "@/components/shared/DataTable";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import i18n from "@/i18n/i18n";

import { AUDIT_TABLE_NAME_DISPLAY, AuditLogEntry, getAuditLogsTableColumns } from "./AuditLogsTableColumns";

interface AuditLogsTableProps {
  data: AuditLogEntry[];
  rowCount: number;
  isLoading: boolean;
  isRefreshing: boolean;
  pagination: PaginationState;
  onPaginationChange: OnChangeFn<PaginationState>;
  columnFilters: ColumnFiltersState;
  onColumnFiltersChange: OnChangeFn<ColumnFiltersState>;
  searchValue?: string;
  onSearchChange?: (value: string) => void;
  onRefresh: () => void;
  onViewLog: (log: AuditLogEntry) => void;
}

const ALL_VALUE = "all";

const ACTION_OPTIONS = [
  { label: "Created", value: "created" },
  { label: "Updated", value: "updated" },
  { label: "Deleted", value: "deleted" },
  { label: "Rotated", value: "rotated" },
  { label: "Kill switch fired", value: "kill_switch_fired" },
] as const;

const TABLE_OPTIONS = [
  { label: "Keys", value: "LiteLLM_VerificationToken" },
  { label: "Teams", value: "LiteLLM_TeamTable" },
  { label: "Users", value: "LiteLLM_UserTable" },
  { label: "Organizations", value: "LiteLLM_OrganizationTable" },
  { label: "Models", value: "LiteLLM_ProxyModelTable" },
  { label: "Agents", value: "LiteLLM_AgentsTable" },
] as const;

const ACTION_FILTER_ITEMS = [
  { value: ALL_VALUE, label: "All Actions" },
  ...ACTION_OPTIONS.map((option) => ({ value: option.value, label: option.label })),
];

const TABLE_FILTER_ITEMS = [
  { value: ALL_VALUE, label: "All Tables" },
  ...TABLE_OPTIONS.map((option) => ({ value: option.value, label: option.label })),
];

const FILTER_LABELS: Record<string, string> = {
  object_id: "Object ID",
  changed_by: "Changed By",
  team_id: "Team ID",
  key_hash: "Key Hash",
  action: "Action",
  table_name: "Table",
};

const formatFilterValue = (columnId: string, value: unknown, t: TFunction = i18n.t.bind(i18n)): string => {
  const raw = String(value);
  if (columnId === "action") {
    return ACTION_OPTIONS.find((option) => option.value === raw)?.label ?? raw;
  }
  if (columnId === "table_name") {
    return Object.hasOwn(AUDIT_TABLE_NAME_DISPLAY, raw) ? t(AUDIT_TABLE_NAME_DISPLAY[raw]) : raw;
  }
  return raw;
};

function AuditLogsEmptyState({ filtered }: { filtered: boolean }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <ScrollText className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">
        {filtered ? t("logs.audit.table.emptyFilteredTitle") : t("logs.audit.table.emptyTitle")}
      </div>
      <div className="max-w-xs text-center text-sm text-muted-foreground">
        {filtered ? t("logs.audit.table.emptyFilteredDescription") : t("logs.audit.table.emptyDescription")}
      </div>
    </div>
  );
}

export function AuditLogsTable({
  data,
  rowCount,
  isLoading,
  isRefreshing,
  pagination,
  onPaginationChange,
  columnFilters,
  onColumnFiltersChange,
  searchValue,
  onSearchChange,
  onRefresh,
  onViewLog,
}: AuditLogsTableProps) {
  const { t } = useTranslation();
  const [filtersOpen, setFiltersOpen] = useState(false);
  const columns = useMemo(() => getAuditLogsTableColumns({ onViewLog }, t), [onViewLog, t]);
  const hasActiveSearch = Boolean(searchValue?.trim());

  return (
    <DataTable
      fillHeight
      data={data}
      columns={columns}
      getRowId={(row) => row.id}
      paginationMode="server"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      rowCount={rowCount}
      filterMode="server"
      columnFilters={columnFilters}
      onColumnFiltersChange={onColumnFiltersChange}
      isLoading={isLoading}
      loadingMessage={t("logs.audit.table.loading")}
      noDataMessage={<AuditLogsEmptyState filtered={columnFilters.length > 0 || hasActiveSearch} />}
      size="compact"
      toolbar={(table) => (
        <>
          <DataTableToolbar
            table={table}
            searchValue={searchValue}
            onSearchChange={onSearchChange}
            searchPlaceholder={t("logs.audit.table.searchPlaceholder")}
            onRefresh={onRefresh}
            isRefreshing={isRefreshing}
            onOpenFilters={() => setFiltersOpen(true)}
            filterLabels={FILTER_LABELS}
            formatFilterValue={formatFilterValue}
            showViewOptions={false}
          />
          <DataTableFilterDrawer
            table={table}
            open={filtersOpen}
            onOpenChange={setFiltersOpen}
            title={t("logs.audit.table.filtersTitle")}
            description={t("logs.audit.table.filtersDescription")}
          >
            {({ get, set }) => (
              <>
                <DataTableFilterField label={t("logs.audit.table.objectId")}>
                  <Input
                    value={(get("object_id") as string) ?? ""}
                    onChange={(event) => set("object_id", event.target.value)}
                    placeholder={t("logs.audit.table.objectIdPlaceholder")}
                  />
                </DataTableFilterField>
                <DataTableFilterField label={t("logs.audit.table.changedBy")}>
                  <Input
                    value={(get("changed_by") as string) ?? ""}
                    onChange={(event) => set("changed_by", event.target.value)}
                    placeholder={t("logs.audit.table.changedByPlaceholder")}
                  />
                </DataTableFilterField>
                <DataTableFilterField label={t("logs.audit.table.teamId")}>
                  <Input
                    value={(get("team_id") as string) ?? ""}
                    onChange={(event) => set("team_id", event.target.value)}
                    placeholder={t("logs.audit.table.teamIdPlaceholder")}
                  />
                </DataTableFilterField>
                <DataTableFilterField label={t("logs.audit.table.keyHash")}>
                  <Input
                    value={(get("key_hash") as string) ?? ""}
                    onChange={(event) => set("key_hash", event.target.value)}
                    placeholder={t("logs.audit.table.keyHashPlaceholder")}
                  />
                </DataTableFilterField>
                <DataTableFilterField label={t("logs.audit.table.action")}>
                  <Select
                    items={ACTION_FILTER_ITEMS}
                    value={(get("action") as string) ?? ALL_VALUE}
                    onValueChange={(value) => set("action", value === ALL_VALUE ? undefined : value)}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder={t("logs.audit.table.allActions")} />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={ALL_VALUE}>{t("logs.audit.table.allActions")}</SelectItem>
                      {ACTION_OPTIONS.map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </DataTableFilterField>
                <DataTableFilterField label={t("logs.audit.table.table")}>
                  <Select
                    items={TABLE_FILTER_ITEMS}
                    value={(get("table_name") as string) ?? ALL_VALUE}
                    onValueChange={(value) => set("table_name", value === ALL_VALUE ? undefined : value)}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder={t("logs.audit.table.allTables")} />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={ALL_VALUE}>{t("logs.audit.table.allTables")}</SelectItem>
                      {TABLE_OPTIONS.map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </DataTableFilterField>
              </>
            )}
          </DataTableFilterDrawer>
        </>
      )}
    />
  );
}
