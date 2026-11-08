import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";
import { getProxyBaseUrl } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { backgroundFetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";

type Monitoring = components["schemas"]["PerformanceMonitoring"];
function monitoringCopy({ enabled, alerts_used: used, alert_limit: limit }: Monitoring) {
  if (limit !== null && used >= limit)
    return [
      "Performance monitoring paused",
      `All ${limit} included alerts have been detected. Saved alerts remain available.`,
    ];
  if (!enabled)
    return [
      "Performance alerts",
      `Detect unusual spend, latency, errors, and traffic.${limit === null ? "" : ` ${limit - used} included alerts remaining.`}`,
    ];
  return [
    "Performance alerts enabled",
    `Checks run hourly on completed days.${limit === null ? "" : ` ${used} of ${limit} included alerts detected.`}`,
  ];
}

export default function PerformanceMonitoring({
  accessToken,
  readOnly = false,
  settings = false,
}: {
  accessToken: string | null;
  readOnly?: boolean;
  settings?: boolean;
}) {
  const { data: user } = useCurrentUser(accessToken);
  const client = useQueryClient();
  const options = {
    queryKey: ["performanceMonitoring", getProxyBaseUrl(), user?.user_id, user?.user_role],
    enabled: Boolean(accessToken && user?.user_id),
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const { data } = await backgroundFetchClient.GET("/alerts/monitoring", { signal });
      if (!data) throw new Error("Could not load Performance monitoring");
      return data;
    },
    retry: false,
    refetchInterval: 5 * 60_000,
  };
  const query = useQuery(options);
  const update = useMutation({
    mutationFn: async (enabled: boolean) => {
      const { data } = await backgroundFetchClient.PATCH("/alerts/monitoring", { body: { enabled } });
      if (!data) throw new Error("Could not update Performance monitoring");
      return data;
    },
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["performanceMonitoring"] });
      await client.invalidateQueries({ queryKey: ["issueFamilies"] });
    },
  });
  if (query.isPending) return null;
  if (query.isError)
    return (
      <div role="alert" className="flex flex-wrap items-center gap-2 text-sm">
        <p>Could not load Performance monitoring.</p>
        <Button size="sm" variant="outline" onClick={() => void query.refetch()}>
          Retry
        </Button>
      </div>
    );
  return (
    <MonitoringPanel
      state={query.data}
      readOnly={readOnly}
      settings={settings}
      pending={update.isPending}
      error={update.error}
      onUpdate={(enabled) => update.mutate(enabled)}
    />
  );
}

function MonitoringPanel({
  state,
  readOnly,
  settings,
  pending,
  error,
  onUpdate,
}: {
  state: Monitoring;
  readOnly: boolean;
  settings: boolean;
  pending: boolean;
  error: Error | null;
  onUpdate: (enabled: boolean) => void;
}) {
  const { enabled, alerts_used: used, alert_limit: limit } = state;
  const exhausted = limit !== null && used >= limit;
  if (enabled && limit === null && !settings) return null;
  const [title, description] = monitoringCopy(state);
  const showHandoff = !exhausted && readOnly && !enabled;
  const canManage = !readOnly && (enabled || !exhausted);
  const showToggle = canManage && (!enabled || settings);
  const action = enabled ? "Pause monitoring" : "Enable performance alerts";
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border p-3">
      <div className="min-w-0 space-y-1">
        <p className="text-sm font-medium">{title}</p>
        <p className="text-xs text-muted-foreground">{description}</p>
        {!enabled && <p className="text-xs text-muted-foreground">Model-deprecation alerts are unaffected.</p>}
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error.message}
          </p>
        )}
      </div>
      {exhausted && (
        <Button size="sm" variant="outline" render={<a href="https://www.litellm.ai/enterprise" />}>
          Upgrade to Enterprise
        </Button>
      )}
      {showHandoff && <span className="text-xs text-muted-foreground">An admin can enable monitoring.</span>}
      {showToggle && (
        <Button
          size="sm"
          variant={enabled ? "outline" : "default"}
          disabled={pending}
          onClick={() => onUpdate(!enabled)}
        >
          {pending ? "Saving…" : action}
        </Button>
      )}
    </div>
  );
}
