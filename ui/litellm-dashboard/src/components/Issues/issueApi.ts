import { useQuery } from "@tanstack/react-query";
import { useCurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";
import { backgroundFetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { getProxyBaseUrl } from "@/components/networking";
import { hasCapability } from "@/utils/capabilities";
import { uiHref } from "@/utils/uiHref";
import { formatAlertValue } from "@/components/UsageAlerts/usageAlertApi";

export type Alert = components["schemas"]["Alert"];
export type IssuePage = components["schemas"]["AlertList"];
export type AlertFamily = Alert["family"];
export type IssueStatus = Alert["status"];
export type AlertComparison = components["schemas"]["AlertComparison"];
export const ISSUE_FAMILIES = ["retirement", "latency", "reliability", "spend", "traffic"] as const;
export const ISSUE_STATUSES = ["open", "resolved", "dismissed"] as const;
export const FAMILY_LABELS = {
  retirement: "Model retirement",
  latency: "Latency regression",
  reliability: "Reliability",
  spend: "Unexpected spend",
  traffic: "Traffic change",
};
const SCOPE_LABELS = {
  deployment: "Gateway",
  key: "Key",
  team: "Team",
  user: "User",
  model: "Model",
  provider: "Provider",
};
const PRIMARY_METRICS = {
  retirement: null,
  latency: "latency",
  reliability: "error_rate",
  spend: "spend",
  traffic: "requests",
};

export const issueHref = (id: string) => uiHref(`alerts?alert=${encodeURIComponent(id)}`);
export const issuesHref = () => uiHref("alerts");

export function issueResource(issue: Alert): string {
  const { scope_type: scope, scope_id: id } = issue.cohort;
  const label = SCOPE_LABELS[scope];
  if (scope === "deployment" && issue.scope_label === label) return label;
  if (issue.scope_label) return `${label} · ${issue.scope_label}`;
  if (!id) return label;
  const short = id.length > 48 ? `${id.slice(0, 24)}…${id.slice(-8)}` : id;
  return `${label} · ${short}`;
}

export const primaryComparison = (issue: Alert) =>
  (issue.comparisons ?? []).find((comparison) => comparison.metric === PRIMARY_METRICS[issue.family]) ??
  issue.comparisons?.[0];

const READABLE_METRICS = {
  spend: "Spend on",
  requests: "Requests to",
  latency: "Response time for",
  errors: "Errors for",
  error_rate: "Error rate for",
};

export function issueHeadline(issue: Alert): string {
  const comparison = primaryComparison(issue);
  if (issue.deprecation || !comparison) return issue.title;
  const { metric, observed_value: actual, baseline_value: baseline, direction } = comparison;
  const label = issue.scope_label ?? issue.cohort.scope_id ?? "the gateway";
  const subject = issue.cohort.scope_type === "provider" ? `${label} models` : label;
  const prefix = `${READABLE_METRICS[metric]} ${subject}`;
  if (baseline <= 0)
    return `${prefix} rose from ${formatAlertValue(metric, baseline)} to ${formatAlertValue(metric, actual)}`;
  if (direction === "decrease") {
    const percent = ((1 - actual / baseline) * 100).toLocaleString(undefined, { maximumFractionDigits: 1 });
    return `${prefix} fell ${percent}%`;
  }
  const ratio = (actual / baseline).toLocaleString(undefined, { maximumFractionDigits: 1 });
  return `${prefix} rose to ${ratio}× normal`;
}

export function issueComparisonText(issue: Alert): string {
  if (issue.deprecation) {
    const { requests, model } = issue.deprecation;
    const replacement = model.successor_model ? ` Suggested replacement: ${model.successor_model}.` : "";
    return `${requests.toLocaleString()} recent requests.${replacement}`;
  }
  const comparison = primaryComparison(issue);
  if (!comparison) return "Open the alert to review its recorded activity.";
  const actual = formatAlertValue(comparison.metric, comparison.observed_value);
  const baseline = formatAlertValue(comparison.metric, comparison.baseline_value);
  return `${actual} on ${issueDate(issue.observed_on)} UTC, compared with the usual ${baseline}.`;
}

export function issueImpactText(issue: Alert): string {
  const impact = issue.impact ?? {};
  return (["keys", "teams", "models"] as const)
    .map((field) => {
      const count = impact[field];
      if (count === null || count === undefined) return "";
      return `${count.toLocaleString()} ${count === 1 ? field.slice(0, -1) : field}`;
    })
    .filter(Boolean)
    .join(" · ");
}

const DATE_OPTIONS: Intl.DateTimeFormatOptions = { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" };
export const issueDate = (value: string) => new Date(value).toLocaleDateString(undefined, DATE_OPTIONS);

interface IssueListOptions {
  status?: IssueStatus;
  family?: AlertFamily;
  kind?: Alert["kind"];
  limit?: number;
  offset?: number;
}

export function useIssues(accessToken: string | null, listOptions: IssueListOptions = {}) {
  const { data: user } = useCurrentUser(accessToken);
  const { status = "open", family, kind, limit = 25, offset = 0 } = listOptions;
  const enabled = Boolean(accessToken && user?.user_id && hasCapability(user.user_role, "viewGlobalSpend"));
  const options = {
    queryKey: [
      "issues",
      getProxyBaseUrl(),
      user?.user_id,
      user?.user_role,
      "list",
      status,
      family,
      kind,
      limit,
      offset,
    ],
    enabled,
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const { data } = await backgroundFetchClient.GET("/alerts", {
        params: { query: { status, family, kind, limit, offset } },
        signal,
      });
      if (!data) throw new Error("Could not load alerts");
      return data;
    },
    staleTime: 60_000,
    refetchInterval: 5 * 60_000,
    retry: false,
  };
  const query = useQuery(options);
  return { ...query, enabled, data: enabled ? query.data : undefined };
}

export function useIssue(accessToken: string | null, id: string) {
  const { data: user } = useCurrentUser(accessToken);
  const enabled = Boolean(accessToken && user?.user_id);
  const options = {
    queryKey: ["issues", getProxyBaseUrl(), user?.user_id, user?.user_role, "detail", id],
    enabled,
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const { data } = await backgroundFetchClient.GET("/alerts/{alert_id}", {
        params: { path: { alert_id: id } },
        signal,
      });
      if (!data) throw new Error("Could not load this alert");
      return data;
    },
    staleTime: 60_000,
    retry: false,
  };
  const query = useQuery(options);
  return { ...query, enabled, data: enabled && !query.isError ? query.data : undefined };
}

export async function updateIssueStatus(issue: Alert, status: IssueStatus) {
  const { data } = await backgroundFetchClient.PATCH("/alerts/{alert_id}", {
    params: { path: { alert_id: issue.id } },
    body: { status, expected_status: issue.status },
  });
  if (!data) throw new Error("Could not update this alert");
  return data;
}
