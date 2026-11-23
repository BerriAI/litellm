import { useMemo } from "react";
import { skipToken, useQuery } from "@tanstack/react-query";
import { parseAsString, useQueryStates } from "nuqs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { overviewTotals } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import type { DailyData } from "@/components/UsagePage/types";
import { customerHierarchyUsageCall, type CustomerUsageScope } from "./customerUsageApi";
import { customerUsageAnalysisParsers } from "./CustomerUsageAnalysis";
import { customerAppParsers } from "./CustomerHierarchyNavigation";
import {
  calendarDate,
  customerPresetRange,
  customerDateSelection,
  customerDatePatch,
  customerUsageDateParsers,
} from "./customerUsageRange";

const usageParsers = {
  ...customerAppParsers,
  org: parseAsString,
  team: parseAsString,
  member: parseAsString,
  ...customerUsageAnalysisParsers,
  ...customerUsageDateParsers,
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
  const { dateValue, preset } = useMemo(() => customerDateSelection(params), [params]);
  const startDate = calendarDate(dateValue.from);
  const endDate = calendarDate(dateValue.to);
  const scope = params.org ? { org: params.org, team: params.team, member: params.member } : null;
  const baseRequest = usageRequest(scope, accessToken, startDate, endDate);
  const request = baseRequest ? { ...baseRequest, appId: params.app_id } : null;
  const month = customerPresetRange("month");
  const monthStartDate = calendarDate(month.from);
  const monthEndDate = calendarDate(month.to);
  const year = customerPresetRange("YTD");
  const yearStartDate = calendarDate(year.from);
  const yearEndDate = calendarDate(year.to);
  const usage = useQuery({
    queryKey: [
      "customer-hierarchy",
      "usage",
      params.org,
      params.team,
      params.member,
      params.app_id,
      startDate,
      endDate,
      monthStartDate,
      yearEndDate,
    ],
    queryFn: request
      ? ({ signal }) => {
          const periodRequest = { ...request, monthStartDate, monthEndDate, yearStartDate, yearEndDate, signal };
          return customerHierarchyUsageCall(periodRequest);
        }
      : skipToken,
  });
  const data = usage.data;
  const results = data?.results ?? EMPTY_RESULTS;
  const totals = useMemo(() => overviewTotals(data?.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA, null), [data]);
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
    const patch = customerDatePatch(value, selectedPreset);
    if (patch) void setParams({ ...patch, usagePage: null }, { history: "push" });
  };
  const selectOrganization = () => void setParams(ORGANIZATION_PATCH, { history: "push" });
  const selectTeam = () => void setParams({ ...TEAM_PATCH, team: data?.team?.id ?? params.team }, { history: "push" });
  const setApp = (app_id: string | null) =>
    void setParams({ app_id, search: null, usagePage: null }, { history: "push" });

  return {
    params,
    scope,
    dateValue,
    preset,
    usage,
    data,
    results,
    totals,
    drillDown,
    setDateRange,
    selectOrganization,
    selectTeam,
    setApp,
    appOrganizationSpend: params.app_id ? null : data?.monthlyOrganizationSpend ?? null,
  };
}
