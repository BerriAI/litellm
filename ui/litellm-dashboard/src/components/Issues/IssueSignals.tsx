import { useQuery } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { getProxyBaseUrl } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { backgroundFetchClient } from "@/lib/http/api";
import { formatAlertValue, METRIC_LABELS, usageAlertHref } from "@/components/UsageAlerts/usageAlertApi";
import { useQueryState, parseAsInteger } from "nuqs";
import { Issue } from "./issueApi";

export default function IssueSignals({ issue }: { issue: Issue }) {
  const { userId, userRole, isAuthorized, accessToken } = useAuthorized();
  const [page, setPage] = useQueryState("signal_page", parseAsInteger.withDefault(1));
  const offset = (Math.max(1, page) - 1) * 20;
  const options = {
    queryKey: ["issueSignals", getProxyBaseUrl(), userId, userRole, issue.id, offset],
    enabled: Boolean(isAuthorized && accessToken),
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const { data } = await backgroundFetchClient.GET("/issues/{issue_id}/signals", {
        params: { path: { issue_id: issue.id }, query: { limit: 20, offset } },
        signal,
      });
      if (!data) throw new Error("Could not load source signals");
      return data;
    },
    retry: false,
  };
  const query = useQuery(options);
  if (!isAuthorized) return null;
  if (query.isPending)
    return (
      <p role="status" className="text-sm text-muted-foreground">
        Loading source signals…
      </p>
    );
  if (query.isError)
    return (
      <div role="alert">
        <p>Could not load source signals.</p>
        <Button variant="outline" onClick={() => void query.refetch()}>
          Retry
        </Button>
      </div>
    );
  return (
    <div className="space-y-3 text-sm">
      <p className="text-xs text-muted-foreground">
        Recorded observations grouped into this issue. Each rule keeps its own baseline and condition.
      </p>
      <ul className="divide-y">
        {query.data.data.map((signal) => {
          const alert = signal.usage_alert;
          return (
            <li key={`${signal.source}-${signal.source_id}`} className="py-3">
              {alert ? (
                <>
                  <a className="font-medium text-primary hover:underline" href={usageAlertHref(alert)}>
                    {alert.rule_name}
                  </a>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {METRIC_LABELS[alert.metric]}: {formatAlertValue(alert.metric, alert.observed_value)} vs{" "}
                    {formatAlertValue(alert.metric, alert.baseline_value)} · {alert.day} UTC
                  </p>
                </>
              ) : (
                <p>Model retirement · {signal.deprecation?.usage_model}</p>
              )}
            </li>
          );
        })}
      </ul>
      {query.data.total === 0 && <p className="text-muted-foreground">No source signals are available.</p>}
      <div className="flex items-center justify-end gap-2">
        <Button variant="outline" size="sm" disabled={offset === 0} onClick={() => void setPage(Math.max(1, page - 1))}>
          Previous
        </Button>
        <span className="text-xs">
          {query.data.total === 0 ? 0 : offset + 1}–{Math.min(offset + 20, query.data.total)} of {query.data.total}
        </span>
        <Button
          variant="outline"
          size="sm"
          disabled={offset + 20 >= query.data.total}
          onClick={() => void setPage(page + 1)}
        >
          Next
        </Button>
      </div>
    </div>
  );
}
