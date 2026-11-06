"use client";

import { ColumnFiltersState, OnChangeFn, PaginationState } from "@tanstack/react-table";
import type { ParseKeys, TFunction } from "i18next";
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

interface FilterOption {
  labelKey: ParseKeys;
  value: string;
}

const ACTION_OPTIONS: readonly FilterOption[] = [
  { labelKey: "logs.audit.actionOptions.created", value: "created" },
  { labelKey: "logs.audit.actionOptions.updated", value: "updated" },
  { labelKey: "logs.audit.actionOptions.deleted", value: "deleted" },
  { labelKey: "logs.audit.actionOptions.rotated", value: "rotated" },
  { labelKey: "logs.audit.actionOptions.killSwitchFired", value: "kill_switch_fired" },
];

const TABLE_OPTIONS: readonly FilterOption[] = [
  { labelKey: "logs.audit.tableNames.keys", value: "LiteLLM_VerificationToken" },
  { labelKey: "logs.audit.tableNames.teams", value: "LiteLLM_TeamTable" },
  { labelKey: "logs.audit.tableNames.users", value: "LiteLLM_UserTable" },
  { labelKey: "logs.audit.tableNames.organizations", value: "LiteLLM_OrganizationTable" },
  { labelKey: "logs.audit.tableNames.models", value: "LiteLLM_ProxyModelTable" },
  { labelKey: "logs.audit.tableNames.agents", value: "LiteLLM_AgentsTable" },
];

const ACTION_FILTER_ITEMS: readonly FilterOption[] = [
  { value: ALL_VALUE, labelKey: "logs.audit.table.allActions" },
  ...ACTION_OPTIONS,
];

const TABLE_FILTER_ITEMS: readonly FilterOption[] = [
  { value: ALL_VALUE, labelKey: "logs.audit.table.allTables" },
  ...TABLE_OPTIONS,
];

const FILTER_LABELS: Record<string, ParseKeys> = {
  object_id: "logs.audit.table.objectId",
  changed_by: "logs.audit.table.changedBy",
  team_id: "logs.audit.table.teamId",
  key_hash: "logs.audit.table.keyHash",
  action: "logs.audit.table.action",
  table_name: "logs.audit.table.table",
};

const formatFilterValue = (columnId: string, value: unknown, t: TFunction = i18n.t.bind(i18n)): string => {
  const raw = String(value);
  if (columnId === "action") {
    const action = ACTION_OPTIONS.find((option) => option.value === raw);
    return action ? t(action.labelKey) : raw;
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
  const filterLabels = useMemo(
    () => Object.fromEntries(Object.entries(FILTER_LABELS).map(([id, labelKey]) => [id, t(labelKey)])),
    [t],
  );
  const withLabel = ({ value, labelKey }: FilterOption) => ({ value, label: t(labelKey) });
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
            filterLabels={filterLabels}
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
                    items={ACTION_FILTER_ITEMS.map(withLabel)}
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
                          {t(option.labelKey)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </DataTableFilterField>
                <DataTableFilterField label={t("logs.audit.table.table")}>
                  <Select
                    items={TABLE_FILTER_ITEMS.map(withLabel)}
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
                          {t(option.labelKey)}
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
