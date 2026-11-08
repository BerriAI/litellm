import { useQueryStates } from "nuqs";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { RefreshCw } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { DataTablePagination, useUrlTableState } from "@/components/shared/DataTable";
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
  issueHeadline,
  issueHref,
  issueImpactText,
  useIssues,
} from "./issueApi";
import { issueRouteParsers } from "./issueRoute";
import PerformanceMonitoring from "./PerformanceMonitoring";
import IssueHistoryChart from "./IssueHistoryChart";
import IssueStatusActions from "./IssueStatusActions";

const TABLE_STATE = { sortFields: [], defaultSort: { id: "", desc: false }, defaultPageSize: 25, filterColumns: [] };
function AlertEntry({ issue }: { issue: Issue }) {
  const impact = issueImpactText(issue);
  return (
    <li className="grid gap-4 px-4 py-5 sm:px-5 lg:grid-cols-[minmax(0,1fr)_220px] lg:gap-8">
      <div className="min-w-0 space-y-3">
        <a
          href={issueHref(issue.id)}
          aria-labelledby={`alert-${issue.id}`}
          className="block space-y-1.5 rounded-sm focus-visible:outline-2 focus-visible:outline-ring"
        >
          <h2 id={`alert-${issue.id}`} className="break-words text-base font-medium leading-6 hover:underline">
            {issueHeadline(issue)}
          </h2>
          <p className="break-words text-sm leading-6 text-muted-foreground">{issueComparisonText(issue)}</p>
          {impact && <p className="text-xs text-muted-foreground">{impact}</p>}
        </a>
        <IssueStatusActions issue={issue} globalAccess />
      </div>
      <div className="hidden self-center lg:block">
        <IssueHistoryChart issue={issue} compact />
      </div>
    </li>
  );
}

export default function IssuesInbox() {
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  const client = useQueryClient();
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
  const refresh = () => {
    void query.refetch();
    void client.invalidateQueries({ queryKey: ["performanceMonitoring"] });
  };
  const { data: alerts = { data: [], total: 0 } } = query;
  const lastPage = Math.max(0, Math.ceil(alerts.total / table.pagination.pageSize) - 1);
  const { pageIndex, pageSize } = table.pagination;
  const { onPaginationChange } = table;
  useEffect(() => {
    if (query.data && query.data.limit < pageSize) {
      const offset = pageIndex * pageSize;
      const limit = query.data.limit;
      const alignedSize =
        Array.from({ length: limit }, (_, index) => limit - index).find((size) => offset % size === 0) ?? 1;
      onPaginationChange({ pageIndex: offset / alignedSize, pageSize: alignedSize });
    } else if (query.data && pageIndex > lastPage) onPaginationChange({ pageIndex: lastPage, pageSize });
  }, [query.data, pageIndex, pageSize, lastPage, onPaginationChange]);
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
        <Button variant="outline" onClick={refresh}>
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
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-md border">
          <div className="flex shrink-0 flex-wrap items-center justify-end gap-2 border-b px-4 py-3">
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
            <Button
              variant="outline"
              size="icon-sm"
              aria-label="Refresh alerts"
              disabled={query.isFetching}
              onClick={refresh}
            >
              <RefreshCw className={query.isFetching ? "animate-spin" : undefined} />
            </Button>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">
            {query.isPending && (
              <p role="status" className="p-5 text-sm text-muted-foreground">
                Loading alerts…
              </p>
            )}
            {!query.isPending && alerts.total === 0 && (
              <div className="space-y-1 p-5">
                <p>No {status} alerts.</p>
                <p className="text-xs text-muted-foreground">
                  Daily performance checks need enough recorded usage history.
                </p>
              </div>
            )}
            <ul aria-label="Alerts" className="divide-y">
              {alerts.data.map((issue) => (
                <AlertEntry key={issue.id} issue={issue} />
              ))}
            </ul>
          </div>
          <div className="shrink-0 border-t">
            <DataTablePagination
              page={pageIndex}
              pageSize={pageSize}
              pageSizeOptions={Array.from(new Set([pageSize, 25, 50, 100])).sort((a, b) => a - b)}
              rowCount={alerts.total}
              onPageChange={(pageIndex) => onPaginationChange({ pageIndex, pageSize })}
              onPageSizeChange={(pageSize) => onPaginationChange({ pageIndex: 0, pageSize })}
              isLoading={query.isPending}
            />
          </div>
        </div>
      </PageTabsContent>
    </PageTabs>
  );
}
