"use client";

import { useMemo } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { parseAsStringLiteral, useQueryState } from "nuqs";
import type { ColumnDef } from "@tanstack/react-table";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { DataTable, DataTableToolbar, useUrlTableState } from "@/components/shared/DataTable";
import { PageTabs, PageTabsContent, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import { Button } from "@/components/ui/button";
import { hasCapability } from "@/utils/capabilities";
import {
  AlertStatus,
  formatAlertValue,
  METRIC_LABELS,
  updateAlertStatus,
  UsageAlert,
  usageAlertEntity,
  usageAlertHref,
  useUsageAlerts,
} from "./usageAlertApi";
import UsageRules from "./UsageRules";

const STATUSES = ["open", "resolved", "dismissed"] as const;
const TABLE_STATE = {
  sortFields: [],
  defaultSort: { id: "", desc: false },
  defaultPageSize: 50,
  filterColumns: [],
  keyPrefix: "usage_alerts",
};

export default function UsageAlertsPanel() {
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  const [status, setStatus] = useQueryState("usage_status", parseAsStringLiteral(STATUSES).withDefault("open"));
  const table = useUrlTableState(TABLE_STATE);
  const query = useUsageAlerts(
    accessToken,
    status,
    table.pagination.pageIndex * table.pagination.pageSize,
    table.pagination.pageSize,
  );
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: ({ id, next }: { id: string; next: AlertStatus }) => updateAlertStatus(id, next),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["usageAlerts"] }),
  });
  const columns = useMemo<ColumnDef<UsageAlert>[]>(
    () => [
      {
        id: "entity",
        header: "Entity",
        cell: ({ row }) => (
          <a
            href={usageAlertHref(row.original)}
            title={usageAlertEntity(row.original)}
            className="block max-w-48 truncate font-medium text-primary hover:underline md:max-w-80"
          >
            {usageAlertEntity(row.original)}
          </a>
        ),
      },
      { id: "metric", header: "Metric", cell: ({ row }) => METRIC_LABELS[row.original.metric] },
      { accessorKey: "day", header: "Day (UTC)" },
      {
        id: "actual",
        header: "Actual",
        cell: ({ row }) => (
          <span className="tabular-nums">{formatAlertValue(row.original.metric, row.original.observed_value)}</span>
        ),
      },
      {
        id: "baseline",
        header: "Baseline",
        cell: ({ row }) => (
          <span className="tabular-nums">{formatAlertValue(row.original.metric, row.original.baseline_value)}</span>
        ),
      },
      {
        id: "condition",
        header: "Condition",
        cell: ({ row }) =>
          `${row.original.direction === "increase" ? "At least" : "At most"} ${formatAlertValue(row.original.metric, row.original.threshold)}`,
      },
      {
        id: "actions",
        header: "",
        cell: ({ row }) =>
          isViewOnly ? null : (
            <div className="flex gap-1">
              {row.original.status === "open" ? (
                <>
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={mutation.isPending}
                    onClick={() => mutation.mutate({ id: row.original.id, next: "resolved" })}
                  >
                    Resolve
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={mutation.isPending}
                    onClick={() => mutation.mutate({ id: row.original.id, next: "dismissed" })}
                  >
                    Dismiss
                  </Button>
                </>
              ) : (
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={mutation.isPending}
                  onClick={() => mutation.mutate({ id: row.original.id, next: "open" })}
                >
                  Reopen
                </Button>
              )}
            </div>
          ),
      },
    ],
    [isViewOnly, mutation],
  );
  if (!hasCapability(userRole, "viewGlobalSpend"))
    return <p className="text-sm text-muted-foreground">Gateway admin access is required.</p>;
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs text-muted-foreground">Checks run hourly for completed UTC days.</p>
        {!isViewOnly && <UsageRules accessToken={accessToken} />}
      </div>
      {mutation.isError && (
        <p role="alert" className="text-sm text-destructive">
          Could not update the alert. Try again.
        </p>
      )}
      {query.isError ? (
        <div role="alert">
          <p>Could not load usage alerts.</p>
          <Button variant="outline" onClick={() => void query.refetch()}>
            Retry
          </Button>
        </div>
      ) : (
        <PageTabs
          value={status}
          onValueChange={(value) => {
            void setStatus(value as AlertStatus);
            table.onPaginationChange({ ...table.pagination, pageIndex: 0 });
          }}
          className="min-h-0 gap-3"
        >
          <PageTabsList>
            {STATUSES.map((value) => (
              <PageTabsTrigger key={value} value={value} className="capitalize">
                {value}
              </PageTabsTrigger>
            ))}
          </PageTabsList>
          <PageTabsContent value={status}>
            <DataTable
              columns={columns}
              data={query.data?.data ?? []}
              getRowId={(row) => row.id}
              isLoading={query.isPending}
              loadingMessage="Loading alerts…"
              noDataMessage="No alerts"
              fillHeight
              size="compact"
              paginationMode="server"
              pagination={table.pagination}
              onPaginationChange={table.onPaginationChange}
              rowCount={query.data?.total ?? 0}
              toolbar={(instance) => (
                <DataTableToolbar
                  table={instance}
                  onRefresh={() => void query.refetch()}
                  isRefreshing={query.isFetching}
                  showViewOptions={false}
                />
              )}
            />
          </PageTabsContent>
        </PageTabs>
      )}
    </div>
  );
}
