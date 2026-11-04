"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, ArrowUpRight, Check, KeyRound } from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import { backgroundFetchClient } from "@/lib/http/api";
import { keyDetailHref, modelGroupHref, teamDetailHref } from "@/utils/entityLinks";
import { cn } from "@/lib/cva.config";
import type { DeprecationAlert } from "./deprecationAlerts";
import type { deprecationUsageRange } from "./deprecationAlerts";

export default function DeprecationAlertItem({
  alert,
  dismissed,
  onDismiss,
  scope,
  range,
}: {
  alert: DeprecationAlert;
  dismissed: boolean;
  onDismiss: () => void;
  scope: string;
  range: ReturnType<typeof deprecationUsageRange>;
}) {
  const [showKeys, setShowKeys] = useState(false);
  const keyOptions = {
    queryKey: ["deprecationAlertKeys", scope, alert.usageModel, range],
    enabled: showKeys,
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const query = { ...range, model_group: alert.usageModel, by_model_group: false, limit: 10, timezone: 0 };
      const { data } = await backgroundFetchClient.GET("/user/daily/activity/aggregated/model_top_keys", {
        params: { query },
        signal,
      });
      if (!data) throw new Error("Affected keys are unavailable");
      return data.api_keys;
    },
    staleTime: 5 * 60 * 1000,
    retry: false,
  };
  const keys = useQuery(keyOptions);
  const retired = alert.model.status === "deprecated";
  const title = retired
    ? `${alert.usageModel} is deprecated`
    : `${alert.usageModel} retires in ${alert.model.days_until_deprecation} days`;
  const modelHref = modelGroupHref(alert.model.model_name);
  function renderAffectedKeys() {
    if (keys.isPending) return <p role="status">Loading affected keys…</p>;
    if (keys.isError)
      return (
        <div>
          <p>Could not load affected keys.</p>
          <Button variant="link" size="sm" onClick={() => void keys.refetch()}>
            Retry
          </Button>
        </div>
      );
    if (keys.data?.length)
      return (
        <>
          <p className="mb-2 text-muted-foreground">Top affected keys by spend · Last 30 days</p>
          <ul className="space-y-2">
            {keys.data.map((key) => (
              <li key={key.api_key} className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <a href={keyDetailHref(key.api_key)} className="break-words font-medium underline underline-offset-2">
                    {key.metadata.key_alias || "Unnamed key"}
                  </a>
                  {key.metadata.team_id && (
                    <a
                      href={teamDetailHref(key.metadata.team_id)}
                      className="mt-0.5 block truncate text-muted-foreground underline underline-offset-2"
                    >
                      Team: {key.metadata.team_id}
                    </a>
                  )}
                </div>
                <span className="shrink-0 tabular-nums">{key.metrics.api_requests.toLocaleString()} requests</span>
              </li>
            ))}
          </ul>
        </>
      );
    return <p>No attributable keys found in this period.</p>;
  }
  return (
    <article aria-label={title} className="flex flex-col gap-3">
      <div className="flex items-start gap-2.5">
        <span
          className={cn(
            "mt-0.5 rounded-lg p-1.5",
            retired ? "bg-destructive/10 text-destructive" : "bg-amber-50 text-amber-700",
          )}
        >
          <AlertTriangle className="size-4" aria-hidden />
        </span>
        <div className="min-w-0 flex-1">
          <p className="break-words text-sm font-medium">{title}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            {alert.requests.toLocaleString()} requests in the last 30 days · Last used {alert.lastUsedDate} UTC
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {retired ? "Deprecated" : "Deprecates"} {alert.model.deprecation_date}
            {alert.model.successor_model ? ` · Suggested replacement: ${alert.model.successor_model}` : ""}
          </p>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="outline" size="sm" onClick={() => setShowKeys(!showKeys)} aria-expanded={showKeys}>
          <KeyRound />
          {showKeys ? "Hide affected keys" : "Review affected keys"}
        </Button>
        {modelHref && (
          <a className={cn(buttonVariants({ size: "sm" }))} href={modelHref}>
            Resolve in model settings <ArrowUpRight className="size-3.5" />
          </a>
        )}
        {dismissed ? (
          <span className="flex items-center gap-1 text-xs text-muted-foreground">
            <Check className="size-3" />
            Dismissed
          </span>
        ) : (
          <Button variant="link" size="sm" onClick={onDismiss}>
            Dismiss
          </Button>
        )}
      </div>
      {showKeys && <div className="rounded-lg border p-3 text-xs">{renderAffectedKeys()}</div>}
    </article>
  );
}
