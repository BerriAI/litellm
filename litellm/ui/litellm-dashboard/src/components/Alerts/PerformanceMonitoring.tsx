import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { backgroundFetchClient } from "@/lib/http/api";
import { getProxyBaseUrl } from "@/components/networking";
import type { components } from "@/lib/http/schema";

function statusMessage(data: components["schemas"]["AlertMonitoring"]) {
  if (data.disabled_by_config) return "alerts.monitoring.disabledByConfig";
  if (!data.enabled) return "alerts.monitoring.off";
  if (data.alert_limit === null) return "alerts.monitoring.enterprise";
  if (data.alerts_used >= data.alert_limit) return "alerts.monitoring.exhausted";
  return "alerts.monitoring.allowance";
}

export default function PerformanceMonitoring({ scope, readOnly }: { scope: string; readOnly: boolean }) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const queryKey = ["alert-monitoring", getProxyBaseUrl(), scope];
  const queryOptions = {
    queryKey,
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const { data } = await backgroundFetchClient.GET("/alerts/monitoring", { signal });
      if (!data) throw new Error("Could not load monitoring settings");
      return data;
    },
    staleTime: 60_000,
    retry: false,
  };
  const query = useQuery(queryOptions);
  const mutation = useMutation({
    mutationFn: async (enabled: boolean) => {
      const { data } = await backgroundFetchClient.PATCH("/alerts/monitoring", { body: { enabled } });
      if (!data) throw new Error("Could not save monitoring settings");
      return data;
    },
    onSuccess: async (data) => {
      await client.cancelQueries({ queryKey });
      client.setQueryData(queryKey, data);
    },
  });
  if (query.isError)
    return (
      <div role="alert" className="flex flex-wrap items-center gap-2 text-sm">
        <p>{t("alerts.monitoring.loadError")}</p>
        <Button variant="outline" size="sm" onClick={() => void query.refetch()}>
          {t("alerts.panel.retry")}
        </Button>
      </div>
    );
  if (!query.data)
    return (
      <p role="status" className="text-sm text-muted-foreground">
        {t("alerts.monitoring.loading")}
      </p>
    );
  const data = query.data;
  const messageKey = statusMessage(data);
  const message = t(messageKey, { used: data.alerts_used, limit: data.alert_limit });
  return (
    <div className="space-y-2">
      <div className="space-y-2 text-sm">
        <div className="flex items-center gap-3">
          <label htmlFor="performance-monitoring" className="font-medium">
            {t("alerts.monitoring.title")}
          </label>
          {!readOnly && (
            <Switch
              id="performance-monitoring"
              checked={data.enabled}
              disabled={mutation.isPending || data.disabled_by_config}
              onCheckedChange={(enabled) => mutation.mutate(enabled)}
              aria-describedby="performance-monitoring-status"
            />
          )}
        </div>
        <p id="performance-monitoring-status" className="text-muted-foreground" role="status">
          {message}
          {messageKey === "alerts.monitoring.exhausted" && (
            <a
              className="ml-2 text-foreground underline underline-offset-4"
              href="https://www.litellm.ai/enterprise"
              target="_blank"
              rel="noopener noreferrer"
            >
              {t("alerts.monitoring.upgrade")}
            </a>
          )}
        </p>
      </div>
      {mutation.isError && (
        <p role="alert" className="text-sm text-destructive">
          {t("alerts.monitoring.saveError")}
        </p>
      )}
    </div>
  );
}
