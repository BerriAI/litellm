import { useQueryStates } from "nuqs";
import type { ColumnDef } from "@tanstack/react-table";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { DataTable, DataTableToolbar, useUrlTableState } from "@/components/shared/DataTable";
import { PageTabs, PageTabsContent, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import { Button } from "@/components/ui/button";
import { hasCapability } from "@/utils/capabilities";
import UsageRules from "@/components/UsageAlerts/UsageRules";
import {
  FAMILY_LABELS,
  ISSUE_FAMILIES,
  ISSUE_STATUSES,
  Issue,
  IssueFamily,
  IssueStatus,
  issueComparisonText,
  issueDate,
  issueHref,
  issueImpactText,
  issueResource,
  useIssues,
} from "./issueApi";
import { issueRouteParsers } from "./issueRoute";
import PerformanceMonitoring from "./PerformanceMonitoring";

const TABLE_STATE = { sortFields: [], defaultSort: { id: "", desc: false }, defaultPageSize: 25, filterColumns: [] };
const columns: ColumnDef<Issue>[] = [
  {
    id: "issue",
    header: "Alert",
    meta: { className: "whitespace-normal" },
    cell: ({ row }) => (
      <div className="min-w-0 space-y-1 py-1">
        <a
          href={issueHref(row.original.id)}
          className="block max-w-64 truncate font-medium hover:underline md:max-w-xl"
          title={row.original.title}
        >
          {row.original.title}
        </a>
        <p
          className="max-w-64 truncate text-xs text-muted-foreground md:max-w-xl"
          title={row.original.scope_label ?? row.original.cohort.scope_id ?? "Gateway"}
        >
          {issueResource(row.original)}
        </p>
      </div>
    ),
  },
  {
    id: "comparison",
    header: "Change",
    cell: ({ row }) => (
      <div className="space-y-1">
        <p className="text-sm tabular-nums">{issueComparisonText(row.original)}</p>
        <p className="text-xs text-muted-foreground">{issueDate(row.original.observed_on)} UTC</p>
      </div>
    ),
  },
  {
    id: "impact",
    header: "Resources",
    cell: ({ row }) => <span className="text-xs">{issueImpactText(row.original) || "—"}</span>,
  },
  {
    id: "seen",
    header: "Last detected",
    cell: ({ row }) => <span className="text-xs">{issueDate(row.original.last_seen)} UTC</span>,
  },
];

export default function IssuesInbox() {
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  const [params, setParams] = useQueryStates(issueRouteParsers);
  const table = useUrlTableState(TABLE_STATE);
  const status = params.status ?? params.state ?? params.usage_status ?? "open";
  const family = params.family ?? (params.type === "model-deprecations" ? "retirement" : undefined);
  const kind = params.type === "usage" && !family ? ("performance" as const) : undefined;
  const queryOptions = {
    status,
    family,
    kind,
    offset: table.pagination.pageIndex * table.pagination.pageSize,
    limit: table.pagination.pageSize,
  };
  const query = useIssues(accessToken, queryOptions);
  const changeStatus = (next: IssueStatus) => {
    void setParams({ status: next, state: null, usage_status: null });
    table.onPaginationChange({ ...table.pagination, pageIndex: 0 });
  };
  const changeFamily = (next: string) => {
    void setParams({
      family: next === "all" || next === "performance" ? null : (next as IssueFamily),
      type: next === "performance" ? "usage" : null,
    });
    table.onPaginationChange({ ...table.pagination, pageIndex: 0 });
  };
  if (!hasCapability(userRole, "viewGlobalSpend"))
    return <p className="text-sm text-muted-foreground">Gateway admin access is required.</p>;
  if (query.isError)
    return (
      <div role="alert" className="space-y-2">
        <p>Could not load alerts.</p>
        <Button variant="outline" onClick={() => void query.refetch()}>
          Retry
        </Button>
      </div>
    );
  return (
    <PageTabs
      value={status}
      onValueChange={(next) => changeStatus(next as IssueStatus)}
      className="min-h-0 flex-1 gap-3"
    >
      <PerformanceMonitoring accessToken={accessToken} readOnly={isViewOnly} />
      <PageTabsList>
        {ISSUE_STATUSES.map((value) => (
          <PageTabsTrigger key={value} value={value} className="capitalize">
            {value}
          </PageTabsTrigger>
        ))}
      </PageTabsList>
      <PageTabsContent value={status} className="flex min-h-0 flex-1 flex-col">
        <DataTable
          columns={columns}
          data={query.data?.data ?? []}
          getRowId={(issue) => issue.id}
          fillHeight
          size="compact"
          isLoading={query.isPending}
          loadingMessage="Loading alerts…"
          noDataMessage={
            <div className="space-y-1">
              <p>No {status} alerts.</p>
              <p className="text-xs text-muted-foreground">
                Daily performance checks need enough recorded usage history.
              </p>
            </div>
          }
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
            >
              <select
                aria-label="Alert type"
                value={family ?? kind ?? "all"}
                onChange={(event) => changeFamily(event.target.value)}
                className="h-8 rounded-md border border-input bg-background px-2 text-xs"
              >
                <option value="all">All alert types</option>
                {kind && <option value="performance">All performance types</option>}
                {ISSUE_FAMILIES.map((value) => (
                  <option key={value} value={value}>
                    {FAMILY_LABELS[value]}
                  </option>
                ))}
              </select>
              {!isViewOnly && <UsageRules accessToken={accessToken} />}
            </DataTableToolbar>
          )}
        />
      </PageTabsContent>
    </PageTabs>
  );
}
