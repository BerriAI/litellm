import { Settings } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { uiHref } from "@/utils/uiHref";
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

export default function ManageAlerts({ scope, readOnly }: { scope: string; readOnly: boolean }) {
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
    mutationFn: async (
      body: components["schemas"]["AlertMonitoringUpdate"] | components["schemas"]["ModelDeprecationMonitoringUpdate"],
    ) => {
      const { data } = await backgroundFetchClient.PATCH("/alerts/monitoring", { body });
      if (!data) throw new Error("Could not save monitoring settings");
      return data;
    },
    onSuccess: async (data) => {
      await client.cancelQueries({ queryKey });
      client.setQueryData(queryKey, data);
      await client.invalidateQueries({ queryKey: ["deprecationAlerts"] });
    },
  });
  const data = query.data;
  const messageKey = data ? statusMessage(data) : undefined;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold">{t("alerts.unified.title")}</h1>
        <Dialog>
          <DialogTrigger render={<Button variant="outline" size="sm" />}>
            <Settings className="size-4" aria-hidden />
            {t("alerts.monitoring.manage")}
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{t("alerts.monitoring.manage")}</DialogTitle>
              <DialogDescription>{t("alerts.monitoring.description")}</DialogDescription>
            </DialogHeader>
            {query.isError && (
              <div role="alert" className="space-y-2 text-sm">
                <p>{t("alerts.monitoring.loadError")}</p>
                <Button variant="outline" size="sm" onClick={() => void query.refetch()}>
                  {t("alerts.panel.retry")}
                </Button>
              </div>
            )}
            {!query.isError && !data && (
              <p role="status" className="text-sm text-muted-foreground">
                {t("alerts.monitoring.loading")}
              </p>
            )}
            {!query.isError && data && (
              <div className="divide-y divide-border">
                <div className="flex items-start justify-between gap-6 py-4">
                  <div className="space-y-1">
                    <label htmlFor="model-deprecations-monitoring" className="text-sm font-medium">
                      {t("alerts.monitoring.deprecations")}
                    </label>
                    <p className="text-sm text-muted-foreground">{t("alerts.monitoring.deprecationsDescription")}</p>
                  </div>
                  <Switch
                    id="model-deprecations-monitoring"
                    checked={data.model_deprecations_enabled}
                    disabled={readOnly || mutation.isPending}
                    onCheckedChange={(enabled) => mutation.mutate({ model_deprecations_enabled: enabled })}
                  />
                </div>
                <div className="flex items-start justify-between gap-6 py-4">
                  <div className="space-y-1">
                    <label htmlFor="performance-monitoring" className="text-sm font-medium">
                      {t("alerts.monitoring.title")}
                    </label>
                    <p id="performance-monitoring-status" className="text-sm text-muted-foreground" role="status">
                      {messageKey && t(messageKey, { used: data.alerts_used, limit: data.alert_limit })}
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
                  <Switch
                    id="performance-monitoring"
                    checked={data.enabled}
                    disabled={readOnly || mutation.isPending || data.disabled_by_config}
                    onCheckedChange={(enabled) => mutation.mutate({ enabled })}
                    aria-describedby="performance-monitoring-status"
                  />
                </div>
              </div>
            )}
            {mutation.isError && (
              <p role="alert" className="text-sm text-destructive">
                {t("alerts.monitoring.saveError")}
              </p>
            )}
            {readOnly && <p className="text-sm text-muted-foreground">{t("alerts.monitoring.adminOnly")}</p>}
            <p className="text-sm text-muted-foreground">{t("alerts.monitoring.notifications")}</p>
            <a href={uiHref("logging-and-alerts")} className="text-sm underline underline-offset-4">
              {t("alerts.monitoring.connections")}
            </a>
          </DialogContent>
        </Dialog>
      </div>
      {data && (
        <p className="text-xs text-muted-foreground">
          {t(data.alert_limit === null ? "alerts.monitoring.enterpriseNote" : "alerts.monitoring.freeNote")}
        </p>
      )}
    </div>
  );
}
