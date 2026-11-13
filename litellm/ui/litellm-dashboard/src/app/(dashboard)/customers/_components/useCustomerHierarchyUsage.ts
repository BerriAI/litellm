import { useMemo } from "react";
import { skipToken, useQuery } from "@tanstack/react-query";
import { parseAsInteger, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  dateRangeFromParams,
  dateRangePatch,
  defaultUsageDateRange,
  usageUrlParsers,
} from "@/app/(dashboard)/usage/_components/usageUrlState";
import {
  modelColorMap,
  overviewTotals,
  type Granularity,
  type UsageMetric,
} from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { formatDate } from "@/components/networking";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import type { DailyData } from "@/components/UsagePage/types";
import { customerHierarchyUsageCall, type CustomerUsageScope } from "./customerUsageApi";
import { rankCustomerModels, rankCustomerUsage } from "./customerUsageData";
import { CUSTOMER_USAGE_DIMENSIONS, type CustomerUsageDimension } from "./CustomerUsageRankings";

const usageParsers = {
  org: parseAsString,
  team: parseAsString,
  member: parseAsString,
  metric: parseAsStringLiteral(["spend", "tokens", "requests"] as const).withDefault("spend"),
  breakdown: parseAsStringLiteral(CUSTOMER_USAGE_DIMENSIONS).withDefault("models"),
  bucket: parseAsStringLiteral(["day", "week"] as const).withDefault("day"),
  search: parseAsString.withDefault(""),
  usagePage: parseAsInteger.withDefault(1),
  range: usageUrlParsers.range,
  from: usageUrlParsers.from,
  to: usageUrlParsers.to,
};
const EMPTY_RESULTS: readonly DailyData[] = [];
const ORGANIZATION_PATCH = { team: null, member: null, search: null, usagePage: null };
const TEAM_PATCH = { member: null, search: null, usagePage: null };

const usageRequest = (
  scope: CustomerUsageScope | null,
  accessToken: string | null,
  startDate: string | undefined,
  endDate: string | undefined,
) => {
  if (!scope || !accessToken) return null;
  if (!startDate || !endDate) return null;
  return { scope, accessToken, startDate, endDate };
};

export default function useCustomerHierarchyUsage() {
  const { accessToken } = useAuthorized();
  const [params, setParams] = useQueryStates(usageParsers);
  const defaultRange = useMemo(() => defaultUsageDateRange("month_to_date"), []);
  const dateValue = useMemo(() => dateRangeFromParams(params) ?? defaultRange.value, [params, defaultRange]);
  const preset = params.range ?? (params.from === null && params.to === null ? defaultRange.preset : null);
  const startDate = dateValue.from ? formatDate(dateValue.from) : undefined;
  const endDate = dateValue.to ? formatDate(dateValue.to) : undefined;
  const scope = params.org ? { org: params.org, team: params.team, member: params.member } : null;
  const request = usageRequest(scope, accessToken, startDate, endDate);
  const usage = useQuery({
    queryKey: ["customer-hierarchy", "usage", params.org, params.team, params.member, startDate, endDate],
    queryFn: request ? ({ signal }) => customerHierarchyUsageCall({ ...request, signal }) : skipToken,
  });
  const data = usage.data;
  const results = data?.results ?? EMPTY_RESULTS;
  const totals = useMemo(() => overviewTotals(data?.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA, null), [data]);
  const modelColors = useMemo(() => modelColorMap(results), [results]);
  const rows = useMemo(
    () =>
      params.breakdown === "models"
        ? rankCustomerModels(results, params.metric, totals)
        : rankCustomerUsage(data?.[params.breakdown] ?? [], params.metric, totals),
    [data, results, params.breakdown, params.metric, totals],
  );
  const drillDown = (dimension: "teams" | "members", id: string) => {
    const patch = {
      ...(dimension === "teams" ? { team: id, member: null } : { member: id }),
      search: null,
      usagePage: null,
      breakdown: null,
    };
    void setParams(patch, { history: "push" });
  };
  const setDateRange = (value: DateRangePickerValue, selectedPreset: string | null) => {
    const { range, from, to } = dateRangePatch(value, selectedPreset);
    const patch = { range, from, to, usagePage: null };
    void setParams(patch);
  };
  const setDimension = (breakdown: CustomerUsageDimension) =>
    void setParams({ breakdown, search: null, usagePage: null });
  const selectOrganization = () => void setParams(ORGANIZATION_PATCH, { history: "push" });
  const selectTeam = () => void setParams(TEAM_PATCH, { history: "push" });
  const setMetric = (metric: UsageMetric) => void setParams({ metric, usagePage: null });
  const setBucket = (bucket: Granularity) => void setParams({ bucket });
  const setSearch = (search: string) => void setParams({ search, usagePage: null });
  const setPage = (usagePage: number) => void setParams({ usagePage });

  return {
    params,
    scope,
    dateValue,
    preset,
    usage,
    data,
    results,
    totals,
    modelColors,
    rows,
    drillDown,
    setDateRange,
    setDimension,
    selectOrganization,
    selectTeam,
    setMetric,
    setBucket,
    setSearch,
    setPage,
  };
}
