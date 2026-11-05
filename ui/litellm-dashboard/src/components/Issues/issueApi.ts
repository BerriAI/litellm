import { useQuery } from "@tanstack/react-query";
import { useCurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";
import { backgroundFetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { getProxyBaseUrl } from "@/components/networking";
import { hasCapability } from "@/utils/capabilities";
import { uiHref } from "@/utils/uiHref";
import { formatAlertChange, formatAlertValue } from "@/components/UsageAlerts/usageAlertApi";

export type Issue = components["schemas"]["Issue"];
export type IssuePage = components["schemas"]["IssueList"];
export type IssueFamily = Issue["family"];
export type IssueStatus = Issue["status"];
export type IssueComparison = components["schemas"]["IssueComparison"];
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

export const issueHref = (id: string) => uiHref(`alerts?issue=${encodeURIComponent(id)}`);
export const issuesHref = () => uiHref("alerts");

export function issueResource(issue: Issue): string {
  const { scope_type: scope, scope_id: id } = issue.cohort;
  const label = SCOPE_LABELS[scope];
  if (issue.scope_label) return `${label} · ${issue.scope_label}`;
  if (!id) return label;
  const short = id.length > 48 ? `${id.slice(0, 24)}…${id.slice(-8)}` : id;
  return `${label} · ${short}`;
}

export const primaryComparison = (issue: Issue) =>
  (issue.comparisons ?? []).find((comparison) => comparison.metric === PRIMARY_METRICS[issue.family]) ??
  issue.comparisons?.[0];

export function issueComparisonText(issue: Issue): string {
  if (issue.deprecation) return `Retirement · ${issue.deprecation.model.deprecation_date}`;
  const comparison = primaryComparison(issue);
  if (!comparison) return "Comparison unavailable";
  const actual = formatAlertValue(comparison.metric, comparison.observed_value);
  const baseline = formatAlertValue(comparison.metric, comparison.baseline_value);
  if (comparison.baseline_value === 0) return `${actual} vs ${baseline}`;
  return `${formatAlertChange(comparison)} · ${actual} vs ${baseline}`;
}

export function issueImpactText(issue: Issue): string {
  const impact = issue.impact ?? {};
  const values = [
    impact.keys === null || impact.keys === undefined
      ? ""
      : `${impact.keys.toLocaleString()} ${impact.keys === 1 ? "key" : "keys"}`,
    impact.teams === null || impact.teams === undefined
      ? ""
      : `${impact.teams.toLocaleString()} ${impact.teams === 1 ? "team" : "teams"}`,
    impact.models === null || impact.models === undefined
      ? ""
      : `${impact.models.toLocaleString()} ${impact.models === 1 ? "model" : "models"}`,
  ];
  return values.filter(Boolean).join(" · ");
}

const DATE_OPTIONS: Intl.DateTimeFormatOptions = { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" };
export const issueDate = (value: string) => new Date(value).toLocaleDateString(undefined, DATE_OPTIONS);

interface IssueListOptions {
  status?: IssueStatus;
  family?: IssueFamily;
  kind?: Issue["kind"];
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
      const { data } = await backgroundFetchClient.GET("/issues", {
        params: { query: { status, family, kind, limit, offset } },
        signal,
      });
      if (!data) throw new Error("Could not load issues");
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
  const enabled = Boolean(accessToken && user?.user_id && hasCapability(user.user_role, "viewGlobalSpend"));
  const options = {
    queryKey: ["issues", getProxyBaseUrl(), user?.user_id, user?.user_role, "detail", id],
    enabled,
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const { data } = await backgroundFetchClient.GET("/issues/{issue_id}", {
        params: { path: { issue_id: id } },
        signal,
      });
      if (!data) throw new Error("Could not load this issue");
      return data;
    },
    staleTime: 60_000,
    retry: false,
  };
  const query = useQuery(options);
  return { ...query, enabled, data: enabled ? query.data : undefined };
}

export async function updateIssueStatus(issue: Issue, status: IssueStatus) {
  const { data } = await backgroundFetchClient.PATCH("/issues/{issue_id}", {
    params: { path: { issue_id: issue.id } },
    body: { status, expected_status: issue.status },
  });
  if (!data) throw new Error("Could not update this issue");
  return data;
}
