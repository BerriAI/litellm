import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { MoreHorizontal } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { backgroundFetchClient } from "@/lib/http/api";
import { getProxyBaseUrl } from "@/components/networking";
import { hasCapability } from "@/utils/capabilities";
import {
  AlertGroup,
  AlertStatus,
  formatAlertValue,
  METRIC_LABELS,
  updateAlertStatus,
  UsageAlert,
  usageAlertHref,
} from "./usageAlertApi";

export function AlertActions({
  label,
  status,
  pending,
  onChange,
}: {
  label: string;
  status: AlertStatus;
  pending: boolean;
  onChange: (status: AlertStatus) => void;
}) {
  const actions: AlertStatus[] = status === "open" ? ["resolved", "dismissed"] : ["open"];
  const labels = { resolved: "Resolve", dismissed: "Dismiss", open: "Reopen" };
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <Button
            variant="ghost"
            size="icon"
            className="size-8"
            disabled={pending}
            aria-label={`Actions for ${label}`}
          />
        }
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {actions.map((next) => (
          <DropdownMenuItem key={next} onClick={() => onChange(next)}>
            {labels[next]}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function IncidentEvidence({ alert }: { alert: UsageAlert }) {
  const { isViewOnly } = useAuthorized();
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: (status: AlertStatus) => updateAlertStatus(alert.id, status),
    onSuccess: () =>
      Promise.all([
        queryClient.invalidateQueries({ queryKey: ["usageAlertGroups"] }),
        queryClient.invalidateQueries({ queryKey: ["usageAlerts"] }),
      ]),
  });
  return (
    <div className="space-y-2 py-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <a href={usageAlertHref(alert)} className="text-sm font-medium text-primary hover:underline">
            {METRIC_LABELS[alert.metric]}
          </a>
          <p className="break-words text-xs text-muted-foreground">
            {alert.rule_name} · <span className="capitalize">{alert.status}</span>
          </p>
        </div>
        {!isViewOnly && (
          <AlertActions
            label={`${METRIC_LABELS[alert.metric]} alert`}
            status={alert.status}
            pending={mutation.isPending}
            onChange={(status) => mutation.mutate(status)}
          />
        )}
      </div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs sm:grid-cols-3">
        <div>
          <dt className="text-muted-foreground">Observed</dt>
          <dd className="mt-1 tabular-nums">{formatAlertValue(alert.metric, alert.observed_value)}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Baseline</dt>
          <dd className="mt-1 tabular-nums">{formatAlertValue(alert.metric, alert.baseline_value)}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Alert condition</dt>
          <dd className="mt-1 tabular-nums">
            {alert.direction === "increase" ? "At least" : "At most"} {formatAlertValue(alert.metric, alert.threshold)}
          </dd>
        </div>
      </dl>
      <p className="text-xs text-muted-foreground">
        Baseline window {alert.baseline_start}–{alert.baseline_end} UTC
        {alert.baseline_sample_days ? ` · ${alert.baseline_sample_days} recorded days` : ""}
      </p>
      {mutation.isError && (
        <p role="alert" className="text-xs text-destructive">
          Could not update this alert. Try again.
        </p>
      )}
    </div>
  );
}

function MetricHistory({
  group,
  metric,
  status,
}: {
  group: AlertGroup;
  metric: UsageAlert["metric"];
  status: AlertStatus;
}) {
  const { accessToken, userId, userRole, isAuthorized } = useAuthorized();
  const [offset, setOffset] = useState(0);
  const enabled = Boolean(isAuthorized && accessToken && hasCapability(userRole, "viewGlobalSpend"));
  const queryOptions = {
    queryKey: ["usageAlerts", getProxyBaseUrl(), userId, userRole, "group", group.id, metric, status, offset],
    enabled,
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const { data } = await backgroundFetchClient.GET("/usage/alerts", {
        params: {
          query: {
            scope_type: group.scope_type,
            scope_id: group.scope_id ?? undefined,
            start_date: group.day,
            end_date: group.day,
            metric,
            status,
            limit: 10,
            offset,
          },
        },
        signal,
      });
      if (!data) throw new Error("Could not load matching alerts");
      return data;
    },
    retry: false,
  };
  const query = useQuery(queryOptions);
  const total = query.data?.total;
  const invalidPage = total !== undefined && offset > 0 && offset >= total;
  if (invalidPage && total !== undefined) setOffset(Math.max(0, Math.ceil(total / 10) - 1) * 10);
  if (!enabled) return null;
  if (invalidPage) return <p className="py-3 text-xs text-muted-foreground">Loading matching alerts…</p>;
  if (query.isPending) return <p className="py-3 text-xs text-muted-foreground">Loading matching alerts…</p>;
  if (query.isError)
    return (
      <div role="alert" className="flex items-center gap-2 py-3 text-xs">
        Could not load matching alerts.
        <Button size="sm" variant="ghost" onClick={() => void query.refetch()}>
          Retry
        </Button>
      </div>
    );
  return (
    <div>
      <div className="divide-y">
        {query.data.data.map((alert) => (
          <IncidentEvidence key={alert.id} alert={alert} />
        ))}
      </div>
      {query.data.total > 10 && (
        <div className="flex items-center justify-between gap-2 py-2 text-xs text-muted-foreground">
          {offset + 1}–{Math.min(offset + 10, query.data.total)} of {query.data.total} matching alerts
          <div className="flex gap-1">
            <Button size="sm" variant="ghost" disabled={offset === 0} onClick={() => setOffset(offset - 10)}>
              Previous
            </Button>
            <Button
              size="sm"
              variant="ghost"
              disabled={offset + 10 >= query.data.total}
              onClick={() => setOffset(offset + 10)}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricEvidence({
  group,
  metric,
  status,
}: {
  group: AlertGroup;
  metric: AlertGroup["metrics"][number];
  status: AlertStatus;
}) {
  const [expanded, setExpanded] = useState(false);
  return (
    <section>
      {expanded ? (
        <MetricHistory group={group} metric={metric.metric} status={status} />
      ) : (
        <IncidentEvidence alert={metric.representative} />
      )}
      {metric.incident_count > 1 && (
        <Button variant="link" size="sm" className="mb-2 h-auto p-0 text-xs" onClick={() => setExpanded(!expanded)}>
          {expanded ? "Show summary" : `View ${metric.incident_count} matching rules`}
        </Button>
      )}
    </section>
  );
}

export default function UsageAlertGroupDetails({ group, status }: { group: AlertGroup; status: AlertStatus }) {
  return (
    <div className="min-w-0 border-t bg-muted/20 px-4 py-3 sm:px-12">
      <p className="text-xs leading-relaxed text-muted-foreground">
        Saved comparisons for {group.day} UTC. Each rule uses its own baseline. Errors are failed-request counts.
      </p>
      <div className="divide-y">
        {group.metrics.map((metric) => (
          <MetricEvidence key={metric.metric} group={group} metric={metric} status={status} />
        ))}
      </div>
    </div>
  );
}
