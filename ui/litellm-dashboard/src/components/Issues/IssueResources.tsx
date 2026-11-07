import { useQuery } from "@tanstack/react-query";
import { parseAsStringLiteral, useQueryState } from "nuqs";
import type { ColumnDef } from "@tanstack/react-table";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { DataTable, useUrlTableState } from "@/components/shared/DataTable";
import { PageTabs, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import { Button } from "@/components/ui/button";
import { getProxyBaseUrl } from "@/components/networking";
import { storedKeyId } from "@/components/UsagePage/keyActivityData";
import { backgroundFetchClient } from "@/lib/http/api";
import { entityActivityHref } from "@/utils/entityLinks";
import { formatAlertValue } from "@/components/UsageAlerts/usageAlertApi";
import type { Issue } from "./issueApi";

interface ResourceRow {
  id: string;
  name: string;
  href?: string;
  owner: string;
  requests: number;
  spend: number;
  errors: number;
}
const columns: ColumnDef<ResourceRow>[] = [
  {
    id: "resource",
    header: "Resource",
    cell: ({ row }) => (
      <div className="max-w-56 truncate" title={row.original.id}>
        {row.original.href ? (
          <a href={row.original.href} className="font-medium text-primary hover:underline">
            {row.original.name}
          </a>
        ) : (
          row.original.name
        )}
      </div>
    ),
  },
  {
    id: "owner",
    header: "Owner",
    cell: ({ row }) => (
      <span className="block max-w-52 truncate" title={row.original.owner}>
        {row.original.owner || "—"}
      </span>
    ),
  },
  { id: "requests", header: "Requests", cell: ({ row }) => row.original.requests.toLocaleString() },
  { id: "spend", header: "Spend", cell: ({ row }) => formatAlertValue("spend", row.original.spend) },
  { id: "errors", header: "Failed requests", cell: ({ row }) => row.original.errors.toLocaleString() },
];
export const ISSUE_RESOURCE_TABLE_STATE = {
  sortFields: [],
  defaultSort: { id: "", desc: false },
  defaultPageSize: 20,
  filterColumns: [],
  keyPrefix: "resources",
};

export default function IssueResources({ issue, from, to }: { issue: Issue; from: string; to: string }) {
  const { userId, userRole, isAuthorized, accessToken } = useAuthorized();
  const [requestedResource, setResource] = useQueryState(
    "resources",
    parseAsStringLiteral(["keys", "users"] as const).withDefault("keys"),
  );
  const supportsUsers = issue.cohort.scope_type !== "team";
  const resource = supportsUsers ? requestedResource : "keys";
  const table = useUrlTableState(ISSUE_RESOURCE_TABLE_STATE);
  const offset = table.pagination.pageIndex * table.pagination.pageSize;
  const options = {
    queryKey: [
      "issueResources",
      getProxyBaseUrl(),
      userId,
      userRole,
      issue.id,
      from,
      to,
      resource,
      offset,
      table.pagination.pageSize,
    ],
    enabled: Boolean(isAuthorized && accessToken && from <= to),
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const params = {
        path: { issue_id: issue.id },
        query: { start_date: from, end_date: to, limit: table.pagination.pageSize, offset },
      };
      if (resource === "keys") {
        const { data } = await backgroundFetchClient.GET("/issues/{issue_id}/keys", { params, signal });
        if (!data) throw new Error("Could not load keys");
        const rows: ResourceRow[] = data.api_keys.map((key) => ({
          id: key.api_key,
          name: key.metadata.key_alias || key.api_key,
          href: entityActivityHref("key", storedKeyId(key.api_key), { from, to }),
          owner: key.metadata.user_email || key.metadata.user_id || "",
          requests: key.metrics.api_requests ?? 0,
          spend: key.metrics.spend ?? 0,
          errors: key.metrics.failed_requests ?? 0,
        }));
        return { rows, total: data.total_api_keys };
      }
      const { data } = await backgroundFetchClient.GET("/issues/{issue_id}/users", { params, signal });
      if (!data) throw new Error("Could not load users");
      const rows: ResourceRow[] = data.users.map((user) => ({
        id: user.user_id ?? "unattributed",
        name: user.user_alias || user.user_email || (user.user_id ?? "Unattributed"),
        href: user.user_id ? entityActivityHref("user", user.user_id, { from, to }) : undefined,
        owner: user.user_email ?? "",
        requests: user.api_requests ?? 0,
        spend: user.spend ?? 0,
        errors: user.failed_requests ?? 0,
      }));
      return { rows, total: data.total_users };
    },
    staleTime: 60_000,
    retry: false,
  };
  const query = useQuery(options);
  if (!isAuthorized || from > to) return null;
  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        Resources with recorded usage in this alert’s traffic scope from {from} to {to} UTC. These totals describe
        exposure, not an independent finding for every resource.
      </p>
      {Boolean(issue.cohort.model_pairs?.length) && (
        <details>
          <summary className="cursor-pointer text-sm">
            Models in this alert · {issue.cohort.model_pairs?.length}
          </summary>
          <ul className="mt-2 max-h-48 divide-y overflow-y-auto text-sm">
            {issue.cohort.model_pairs?.map((pair) => (
              <li key={`${pair.provider}/${pair.model}`} className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2">
                <a
                  className="break-all text-primary hover:underline"
                  href={entityActivityHref("model", pair.model, { from, to })}
                >
                  {pair.model}
                </a>
                <span className="text-xs text-muted-foreground">{pair.provider}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
      <PageTabs
        value={resource}
        onValueChange={(value) => {
          void setResource(value as "keys" | "users");
          table.onPaginationChange({ ...table.pagination, pageIndex: 0 });
        }}
      >
        <PageTabsList>
          <PageTabsTrigger value="keys">Keys</PageTabsTrigger>
          {supportsUsers && <PageTabsTrigger value="users">Key owners</PageTabsTrigger>}
        </PageTabsList>
      </PageTabs>
      {query.isError ? (
        <div role="alert">
          <p>Could not load resources.</p>
          <Button variant="outline" size="sm" onClick={() => void query.refetch()}>
            Retry
          </Button>
        </div>
      ) : (
        <DataTable
          columns={columns}
          data={query.data?.rows ?? []}
          getRowId={(row) => row.id}
          isLoading={query.isPending}
          paginationMode="server"
          pagination={table.pagination}
          onPaginationChange={table.onPaginationChange}
          rowCount={query.data?.total ?? 0}
          size="compact"
          noDataMessage="No resources in this date range."
        />
      )}
    </div>
  );
}
