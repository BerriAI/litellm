import { useQuery } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { ModelSection } from "@/components/activity_metrics";
import { getProxyBaseUrl } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { EMPTY_DAILY_ACTIVITY_METADATA, toDailyData } from "@/components/UsagePage/dailyActivityApi";
import { keyDetailFromResponse, overallUsageMetrics } from "@/components/UsagePage/keyActivityData";
import { backgroundFetchClient } from "@/lib/http/api";
import type { Issue } from "./issueApi";

export default function IssueActivity({ issue, from, to }: { issue: Issue; from: string; to: string }) {
  const { userId, userRole, isAuthorized, accessToken } = useAuthorized();
  const enabled = Boolean(isAuthorized && accessToken && from <= to);
  const options = {
    queryKey: ["issueActivity", getProxyBaseUrl(), userId, userRole, issue.id, from, to],
    enabled,
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const { data } = await backgroundFetchClient.GET("/issues/{issue_id}/activity", {
        params: { path: { issue_id: issue.id }, query: { start_date: from, end_date: to } },
        signal,
      });
      if (!data) throw new Error("Could not load activity");
      return data;
    },
    staleTime: 60_000,
    retry: false,
  };
  const query = useQuery(options);
  if (!isAuthorized) return null;
  if (from > to)
    return (
      <p role="alert" className="text-sm text-destructive">
        Choose an end date on or after the start date.
      </p>
    );
  if (query.isPending)
    return (
      <p role="status" className="text-sm text-muted-foreground">
        Loading activity…
      </p>
    );
  if (query.isError)
    return (
      <div role="alert" className="flex items-center gap-2 text-sm">
        <p>Could not load activity.</p>
        <Button variant="outline" size="sm" onClick={() => void query.refetch()}>
          Retry
        </Button>
      </div>
    );
  if (query.data.results.length === 0)
    return <p className="text-sm text-muted-foreground">No recorded activity in this date range.</p>;
  const id = issue.cohort.scope_id ?? issue.id;
  const keyMetrics = issue.cohort.scope_type === "key" ? keyDetailFromResponse(query.data, id, []) : undefined;
  const metrics =
    keyMetrics ?? overallUsageMetrics(toDailyData(query.data), query.data.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA);
  return <ModelSection modelName={id} metrics={metrics} />;
}
