"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { parseAsStringLiteral, useQueryState } from "nuqs";
import type { ColumnDef } from "@tanstack/react-table";
import { ChevronRight } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { DataTable, DataTableToolbar, useUrlTableState } from "@/components/shared/DataTable";
import { PageTabs, PageTabsContent, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import { Button } from "@/components/ui/button";
import { hasCapability } from "@/utils/capabilities";
import {
  AlertGroup,
  AlertStatus,
  formatAlertChange,
  METRIC_SHORT_LABELS,
  SCOPE_LABELS,
  updateAlertGroup,
  UsageAlert,
  usageAlertGroupHref,
  usageAlertGroupName,
  usageAlertGroupTitle,
  usageAlertHref,
  useUsageAlertGroups,
} from "./usageAlertApi";
import UsageAlertGroupDetails, { AlertActions } from "./UsageAlertGroupDetails";
import UsageRules from "./UsageRules";

const STATUSES = ["open", "resolved", "dismissed"] as const;
const SCOPES = ["all", "deployment", "key", "team", "user", "model"] as const;
const TABLE_STATE = {
  sortFields: [],
  defaultSort: { id: "", desc: false },
  defaultPageSize: 25,
  filterColumns: [],
  keyPrefix: "usage_alerts",
};

export default function UsageAlertsPanel() {
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  const [status, setStatus] = useQueryState("usage_status", parseAsStringLiteral(STATUSES).withDefault("open"));
  const [scope, setScope] = useQueryState("usage_scope", parseAsStringLiteral(SCOPES).withDefault("all"));
  const table = useUrlTableState(TABLE_STATE);
  const query = useUsageAlertGroups(accessToken, status, {
    offset: table.pagination.pageIndex * table.pagination.pageSize,
    limit: table.pagination.pageSize,
    scope: scope === "all" ? undefined : scope,
  });
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: ({ group, next }: { group: AlertGroup; next: AlertStatus }) => updateAlertGroup(group, status, next),
    onSuccess: async () => {
      if (query.data?.data.length === 1 && table.pagination.pageIndex > 0) {
        table.onPaginationChange({ ...table.pagination, pageIndex: table.pagination.pageIndex - 1 });
      }
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["usageAlertGroups"] }),
        queryClient.invalidateQueries({ queryKey: ["usageAlerts"] }),
      ]);
    },
  });
  const columns: ColumnDef<AlertGroup>[] = [
    {
      id: "resource",
      header: "Resource",
      meta: { className: "whitespace-normal" },
      cell: ({ row }) => (
        <div className="flex items-start gap-2 py-2">
          <Button
            variant="ghost"
            size="icon"
            className="size-8 shrink-0"
            aria-label={`Details for ${usageAlertGroupName(row.original)}`}
            aria-expanded={row.getIsExpanded()}
            onClick={() => row.toggleExpanded()}
          >
            <ChevronRight className={`size-4 transition-transform ${row.getIsExpanded() ? "rotate-90" : ""}`} />
          </Button>
          <div className="min-w-0 flex-1 space-y-2">
            <div className="min-w-0">
              <a
                href={usageAlertGroupHref(row.original)}
                title={usageAlertGroupTitle(row.original)}
                aria-label={usageAlertGroupTitle(row.original)}
                className="block max-w-48 truncate text-sm font-medium text-primary hover:underline sm:max-w-xl"
              >
                {usageAlertGroupName(row.original)}
              </a>
              <p className="mt-0.5 text-xs text-muted-foreground">
                {SCOPE_LABELS[row.original.scope_type]} · {row.original.day} UTC
              </p>
            </div>
            <div className="flex flex-wrap gap-x-5 gap-y-1.5">
              {row.original.metrics.map((metric) => (
                <a
                  key={metric.metric}
                  href={usageAlertHref(metric.representative)}
                  className="whitespace-nowrap text-xs hover:underline"
                >
                  <span className="text-muted-foreground">{METRIC_SHORT_LABELS[metric.metric]}</span>{" "}
                  <span className="tabular-nums">{formatAlertChange(metric.representative)}</span>
                </a>
              ))}
            </div>
          </div>
        </div>
      ),
    },
    {
      id: "actions",
      header: "",
      size: 40,
      meta: { className: "w-10 text-right align-top pt-3" },
      cell: ({ row }) =>
        isViewOnly ? null : (
          <AlertActions
            label={usageAlertGroupName(row.original)}
            status={status}
            pending={mutation.isPending}
            onChange={(next) => mutation.mutate({ group: row.original, next })}
          />
        ),
    },
  ];
  if (!hasCapability(userRole, "viewGlobalSpend")) {
    return <p className="text-sm text-muted-foreground">Gateway admin access is required.</p>;
  }
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <p className="text-xs text-muted-foreground">Checks use completed UTC days.</p>
      {mutation.isError && (
        <p role="alert" className="text-sm text-destructive">
          Could not update the alerts. Try again.
        </p>
      )}
      {query.isError ? (
        <div role="alert" className="space-y-2">
          <p>Could not load performance alerts.</p>
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
              loadingMessage="Loading performance alerts…"
              noDataMessage={
                <div className="space-y-1">
                  <p>No {status} performance alerts.</p>
                  {status === "open" && (
                    <p className="text-xs text-muted-foreground">
                      Daily comparisons appear after enough usage history is available.
                    </p>
                  )}
                </div>
              }
              fillHeight
              paginationMode="server"
              pagination={table.pagination}
              onPaginationChange={table.onPaginationChange}
              rowCount={query.data?.total ?? 0}
              getRowCanExpand={() => true}
              renderSubComponent={({ row }) => <UsageAlertGroupDetails group={row.original} status={status} />}
              toolbar={(instance) => (
                <DataTableToolbar
                  table={instance}
                  onRefresh={() => void query.refetch()}
                  isRefreshing={query.isFetching}
                  showViewOptions={false}
                >
                  <select
                    aria-label="Resource type"
                    value={scope}
                    className="h-8 rounded-md border border-input bg-background px-2 text-xs"
                    onChange={(event) => {
                      void setScope(event.target.value as UsageAlert["scope_type"] | "all");
                      table.onPaginationChange({ ...table.pagination, pageIndex: 0 });
                    }}
                  >
                    <option value="all">All resources</option>
                    {Object.entries(SCOPE_LABELS).map(([value, label]) => (
                      <option key={value} value={value}>
                        {label}
                      </option>
                    ))}
                  </select>
                  {!isViewOnly && <UsageRules accessToken={accessToken} />}
                </DataTableToolbar>
              )}
            />
          </PageTabsContent>
        </PageTabs>
      )}
    </div>
  );
}
