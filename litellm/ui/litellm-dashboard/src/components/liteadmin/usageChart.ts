import { z } from "zod";
import type { TFunction } from "i18next";
import i18n from "@/i18n/i18n";
import type { AriaAccess } from "./operations";

export const usageChartRequestSchema = z.strictObject({
  source_id: z.string().min(1).max(100),
  view: z.enum(["daily_spend", "model_spend", "provider_spend", "daily_requests", "daily_tokens", "token_breakdown"]),
});

type UsageChartView = z.infer<typeof usageChartRequestSchema>["view"];
const metric = z.number().finite().nonnegative().nullish();
const metricFields = {
  spend: metric,
  api_requests: metric,
  total_tokens: metric,
  prompt_tokens: metric,
  completion_tokens: metric,
  cache_read_input_tokens: metric,
  cache_creation_input_tokens: metric,
};
const metrics = z.object(metricFields);
const usageReportSchema = z.object({
  daily: z.array(metrics.extend({ date: z.iso.date() })).max(367),
  top_models: z.array(metrics.extend({ name: z.string().nullish() })).max(20),
  top_providers: z.array(metrics.extend({ name: z.string().nullish() })).max(20),
});
const queryFields = {
  start_date: z.iso.date(),
  end_date: z.iso.date(),
  user_id: z.string().nullish(),
  team_ids: z.string().nullish(),
  tags: z.string().nullish(),
};
const usageQuerySchema = z.object(queryFields);

type DefaultUsageScope = "gateway" | "allTeams" | "allTags";
type UsageScope =
  | DefaultUsageScope
  | "yours"
  | "permittedTeams"
  | {
      userId: string | null | undefined;
      teamIds: string | null | undefined;
      tags: string | null | undefined;
    };

interface UsageChartContext {
  period: { start: string; end: string };
  scope: UsageScope;
}

export interface UsageChartSource extends UsageChartContext {
  report: z.infer<typeof usageReportSchema>;
}

export interface UsageChart extends UsageChartContext {
  kind: "bar" | "line";
  title: string;
  unit: "usd" | "requests" | "tokens";
  total: number | null;
  data: { label: string; value: number | null }[];
  note?: string;
}

export function usageChartSource(
  value: unknown,
  query: unknown,
  access: AriaAccess,
  defaultScope: DefaultUsageScope = "gateway",
): UsageChartSource | undefined {
  const report = usageReportSchema.safeParse(value);
  const args = usageQuerySchema.safeParse(query);
  if (!report.success || !args.success) return undefined;
  const { start_date, end_date, user_id, team_ids, tags } = args.data;
  const ownScope = defaultScope === "allTeams" ? "permittedTeams" : "yours";
  const filters = user_id || team_ids || tags ? { userId: user_id, teamIds: team_ids, tags } : defaultScope;
  const scope = access.mode === "usage" && access.scope.type !== "global" ? ownScope : filters;
  return { report: report.data, period: { start: start_date, end: end_date }, scope };
}

const toolScopeLabels = {
  gateway: "Gateway usage",
  allTeams: "All teams",
  allTags: "All tags",
  yours: "Your usage",
  permittedTeams: "Your permitted team usage",
} satisfies Record<Extract<UsageScope, string>, string>;

export function formatUsageChartToolContext({ period, scope }: UsageChartContext) {
  const scopeLabel =
    typeof scope === "string"
      ? toolScopeLabels[scope]
      : [
          scope.userId && `User ${scope.userId}`,
          scope.teamIds && `Teams ${scope.teamIds}`,
          scope.tags && `Tags ${scope.tags}`,
        ]
          .filter(Boolean)
          .join(" · ");
  return { scope: scopeLabel, period: `${period.start} to ${period.end} (UTC)` };
}

export function formatUsageChartContext({ period, scope }: UsageChartContext, t: TFunction) {
  const scopeLabel =
    typeof scope === "string"
      ? t(`aria.chart.scopes.${scope}`)
      : [
          scope.userId && t("aria.chart.scopes.user", { id: scope.userId }),
          scope.teamIds && t("aria.chart.scopes.teams", { ids: scope.teamIds }),
          scope.tags && t("aria.chart.scopes.tags", { tags: scope.tags }),
        ]
          .filter(Boolean)
          .join(" · ");
  return { scope: scopeLabel, period: t("aria.chart.period", period) };
}

function knownTotal(values: readonly (number | null | undefined)[]): number | null {
  if (values.length === 0 || values.some((value) => value == null)) return null;
  return values.reduce<number>((total, value) => total + (value ?? 0), 0);
}

function rankedData(source: UsageChartSource, dimension: "top_models" | "top_providers"): UsageChart["data"] {
  const ranked = [...source.report[dimension]].sort((a, b) => (b.spend ?? 0) - (a.spend ?? 0));
  const shown = ranked.slice(0, 6).map((row) => ({ label: row.name || "Unknown", value: row.spend ?? null }));
  const total = knownTotal(source.report.daily.map((day) => day.spend));
  const visibleTotal = knownTotal(shown.map((row) => row.value));
  const remainder = total == null || visibleTotal == null ? null : total - visibleTotal;
  return remainder != null && remainder > 1e-10 ? [...shown, { label: "Other", value: remainder }] : shown;
}

const dailyViews = {
  daily_spend: { title: "Daily spend", metric: "spend", unit: "usd" },
  daily_requests: { title: "Daily requests", metric: "api_requests", unit: "requests" },
  daily_tokens: { title: "Daily tokens", metric: "total_tokens", unit: "tokens" },
} as const;

export function buildUsageChart(source: UsageChartSource, view: UsageChartView): UsageChart | undefined {
  const base = { period: source.period, scope: source.scope };
  if (view === "model_spend" || view === "provider_spend") {
    const data = rankedData(source, view === "model_spend" ? "top_models" : "top_providers");
    if (data.length === 0) return undefined;
    return {
      ...base,
      kind: "bar",
      title: view === "model_spend" ? "Spend by model" : "Spend by provider",
      unit: "usd",
      total: knownTotal(source.report.daily.map((day) => day.spend)),
      data,
      note: data.some((row) => row.label === "Other")
        ? "Up to six leading entries. Other includes remaining recorded spend."
        : undefined,
    };
  }
  if (view === "token_breakdown") {
    if (source.report.daily.length === 0) return undefined;
    const tokenFields = [
      ["Prompt", "prompt_tokens"],
      ["Completion", "completion_tokens"],
      ["Cache read", "cache_read_input_tokens"],
      ["Cache creation", "cache_creation_input_tokens"],
    ] as const;
    return {
      ...base,
      kind: "bar",
      title: "Token breakdown",
      unit: "tokens",
      total: knownTotal(source.report.daily.map((day) => day.total_tokens)),
      data: tokenFields.map(([label, field]) => ({
        label,
        value: knownTotal(source.report.daily.map((day) => day[field])),
      })),
      note: "Cache counts are reported separately. Do not add them to total tokens.",
    };
  }
  const config = dailyViews[view];
  const data = [...source.report.daily]
    .sort((a, b) => a.date.localeCompare(b.date))
    .map((day) => ({ label: day.date, value: day[config.metric] ?? null }));
  if (data.length === 0) return undefined;
  return {
    ...base,
    kind: "line",
    title: config.title,
    unit: config.unit,
    total: knownTotal(data.map((point) => point.value)),
    data,
  };
}

export function formatUsageChartValue(value: number | null, unit: UsageChart["unit"]): string {
  if (value == null) return i18n.t("aria.chart.unavailable");
  return unit === "usd"
    ? value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 6 })
    : value.toLocaleString("en-US");
}
