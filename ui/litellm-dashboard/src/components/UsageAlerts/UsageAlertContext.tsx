import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { backgroundFetchClient } from "@/lib/http/api";
import { getProxyBaseUrl } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { formatAlertValue, METRIC_LABELS, PerformanceSignal } from "./usageAlertApi";

export default function UsageAlertContext({
  signalId,
  scope,
  entityId,
}: {
  signalId: string;
  scope: PerformanceSignal["scope_type"];
  entityId: string | null;
}) {
  const { accessToken, userId, userRole, isAuthorized } = useAuthorized();
  const options: UseQueryOptions<PerformanceSignal> = {
    queryKey: ["performanceSignal", getProxyBaseUrl(), userId, userRole, signalId],
    enabled: Boolean(isAuthorized && accessToken && signalId),
    queryFn: async ({ signal }) => {
      const { data } = await backgroundFetchClient.GET("/alerts/signals/{signal_id}", {
        params: { path: { signal_id: signalId } },
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
  const verb = alert.metric === "requests" || alert.metric === "errors" ? "were" : "was";
  return (
    <p className="text-sm text-muted-foreground">
      {METRIC_LABELS[alert.metric]} {verb} {formatAlertValue(alert.metric, alert.observed_value)} on {alert.day} UTC.
      The baseline was {formatAlertValue(alert.metric, alert.baseline_value)}
      {alert.baseline_sample_days ? ` across ${alert.baseline_sample_days} recorded days` : ""}. The alert condition was{" "}
      {alert.direction === "increase" ? "at least" : "at most"} {formatAlertValue(alert.metric, alert.threshold)}.
    </p>
  );
}
