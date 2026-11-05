import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { backgroundFetchClient } from "@/lib/http/api";
import { getProxyBaseUrl } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { formatAlertValue, METRIC_LABELS, UsageAlert } from "./usageAlertApi";

export default function UsageAlertContext({
  alertId,
  scope,
  entityId,
}: {
  alertId: string;
  scope: UsageAlert["scope_type"];
  entityId: string | null;
}) {
  const { accessToken, userId, userRole, isAuthorized } = useAuthorized();
  const options: UseQueryOptions<UsageAlert> = {
    queryKey: ["usageAlert", getProxyBaseUrl(), userId, userRole, alertId],
    enabled: Boolean(isAuthorized && accessToken && alertId),
    queryFn: async ({ signal }) => {
      const { data } = await backgroundFetchClient.GET("/usage/alerts/{alert_id}", {
        params: { path: { alert_id: alertId } },
        signal,
      });
      if (!data) throw new Error("Alert unavailable");
      return data;
    },
    staleTime: 60_000,
    retry: false,
  };
  const query = useQuery(options);
  if (!isAuthorized) return null;
  if (query.isPending) return <p className="text-sm text-muted-foreground">Loading alert…</p>;
  if (query.isError)
    return (
      <div className="flex items-center gap-2 text-sm" role="alert">
        <span>Could not load this alert.</span>
        <Button size="sm" variant="ghost" onClick={() => void query.refetch()}>
          Retry
        </Button>
      </div>
    );
  const alert = query.data;
  if (alert.scope_type !== scope || alert.scope_id !== entityId)
    return <p className="text-sm text-muted-foreground">This alert belongs to a different entity.</p>;
  return (
    <p className="text-sm text-muted-foreground">
      {METRIC_LABELS[alert.metric]} was {formatAlertValue(alert.metric, alert.observed_value)} on {alert.day} UTC. The
      baseline was {formatAlertValue(alert.metric, alert.baseline_value)}
      {alert.baseline_sample_days ? ` across ${alert.baseline_sample_days} recorded days` : ""}.
    </p>
  );
}
