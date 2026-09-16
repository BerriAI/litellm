import { useCallback, useMemo } from "react";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { ColumnFiltersState } from "@tanstack/react-table";
import { parseAsString, useQueryState } from "nuqs";
import { useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";
import { resolveLogoSrc } from "@/lib/assetPaths";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { uiAuditLogsCall } from "../networking";
import { AuditLogEntry } from "./AuditLogsTableColumns";
import { AuditLogsTable } from "./AuditLogsTable";
import { AuditLogDrawer } from "./AuditLogDrawer/AuditLogDrawer";

interface AuditLogsProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  isActive: boolean;
  premiumUser: boolean;
}

const asset_logos_folder = "/ui/assets/";
const auditLogsPreviewImg = `${asset_logos_folder}audit-logs-preview.png`;

const PAGE_SIZE = 50;

const FILTER_COLUMNS = ["object_id", "changed_by", "team_id", "key_hash", "action", "table_name"] as const;
type FilterColumn = (typeof FILTER_COLUMNS)[number];

const TABLE_STATE_OPTIONS: UrlTableStateOptions<FilterColumn> = {
  sortFields: ["updated_at"],
  defaultSort: { id: "updated_at", desc: true },
  defaultPageSize: PAGE_SIZE,
  filterColumns: FILTER_COLUMNS,
  keyPrefix: "audit_",
  urlKeys: { filter_team_id: "filter_team", filter_table_name: "filter_table" },
};

const appliedFilter = (filters: ColumnFiltersState, column: FilterColumn): string | undefined => {
  const value = filters.find((filter) => filter.id === column)?.value;
  return typeof value === "string" ? value : undefined;
};

interface AuditLogsResponse {
  audit_logs: AuditLogEntry[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

const EMPTY_ROWS: AuditLogEntry[] = [];

export default function AuditLogsPanel({
  userID,
  userRole,
  token,
  accessToken,
  isActive,
  premiumUser,
}: AuditLogsProps) {
  const {
    search: searchInput,
    setSearch,
    pagination,
    onPaginationChange,
    columnFilters,
    onColumnFiltersChange,
  } = useUrlTableState(TABLE_STATE_OPTIONS);
  const [debouncedSearch] = useDebouncedValue(searchInput, { wait: DEBOUNCE_WAIT_MS });
  const [selectedLogId, setSelectedLogId] = useQueryState(
    "audit_log_id",
    parseAsString.withOptions({ history: "push" }),
  );

  const searchTerm = debouncedSearch.trim();

  const canQueryAuditLogs = !!accessToken && !!token && !!userRole && !!userID && isActive && premiumUser;

  const query = useQuery<AuditLogsResponse>({
    queryKey: ["audit_logs", pagination.pageIndex, pagination.pageSize, columnFilters, searchTerm],
    queryFn: async () => {
      if (!accessToken) {
        return { audit_logs: [], total: 0, page: 1, page_size: pagination.pageSize, total_pages: 0 };
      }
      return uiAuditLogsCall({
        accessToken,
        page: pagination.pageIndex + 1,
        page_size: pagination.pageSize,
        params: {
          search: searchTerm || undefined,
          object_id: appliedFilter(columnFilters, "object_id"),
          changed_by: appliedFilter(columnFilters, "changed_by"),
          object_key_hash: appliedFilter(columnFilters, "key_hash"),
          object_team_id: appliedFilter(columnFilters, "team_id"),
          action: appliedFilter(columnFilters, "action"),
          table_name: appliedFilter(columnFilters, "table_name"),
          sort_by: "updated_at",
          sort_order: "desc",
        },
      });
    },
    enabled: canQueryAuditLogs,
    placeholderData: keepPreviousData,
  });

  const rows = query.data?.audit_logs ?? EMPTY_ROWS;
  const selectedLog = useMemo(() => rows.find((log) => log.id === selectedLogId) ?? null, [rows, selectedLogId]);

  const handleViewLog = useCallback((log: AuditLogEntry) => void setSelectedLogId(log.id), [setSelectedLogId]);
  const closeDrawer = useCallback(() => void setSelectedLogId(null), [setSelectedLogId]);

  if (!premiumUser) {
    return (
      <div style={{ textAlign: "center", marginTop: "20px" }}>
        <h1 style={{ display: "block", marginBottom: "10px" }}>✨ Enterprise Feature.</h1>
        <p style={{ display: "block", marginBottom: "10px" }}>
          This is a LiteLLM Enterprise feature, and requires a valid key to use.
        </p>
        <p style={{ display: "block", marginBottom: "20px", fontStyle: "italic" }}>
          Here&apos;s a preview of what Audit Logs offer:
        </p>
        <img
          src={resolveLogoSrc(auditLogsPreviewImg)}
          alt="Audit Logs Preview"
          style={{
            maxWidth: "100%",
            maxHeight: "700px",
            borderRadius: "8px",
            boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
            margin: "0 auto",
          }}
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
      </div>
    );
  }

  return (
    <>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">Audit Logs</h1>
      </div>

      <AuditLogsTable
        data={rows}
        rowCount={query.data?.total ?? 0}
        isLoading={query.isLoading}
        isError={query.isError}
        isRefreshing={query.isFetching}
        pagination={pagination}
        onPaginationChange={onPaginationChange}
        columnFilters={columnFilters}
        onColumnFiltersChange={onColumnFiltersChange}
        searchValue={searchInput}
        onSearchChange={setSearch}
        onRefresh={() => query.refetch()}
        onViewLog={handleViewLog}
      />

      <AuditLogDrawer open={selectedLog !== null} onClose={closeDrawer} log={selectedLog} />
    </>
  );
}
