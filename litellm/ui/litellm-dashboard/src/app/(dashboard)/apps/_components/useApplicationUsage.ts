import { skipToken, useQuery } from "@tanstack/react-query";
import useAuthorized from "../../hooks/useAuthorized";
import { customerAppsUsageCall } from "../../customers/_components/customerUsageApi";
import { calendarDate, customerPresetRange } from "../../customers/_components/customerUsageRange";

export type ApplicationUsageScope = Pick<
  Parameters<typeof customerAppsUsageCall>[2],
  "startDate" | "endDate" | "appGroupId"
>;

export function useApplicationUsage(
  ids: readonly string[],
  includeOverall: boolean,
  dimension: "appId" | "appGroupId" = "appId",
  scope?: ApplicationUsageScope,
) {
  const { accessToken, userId } = useAuthorized();
  const month = customerPresetRange("month");
  const startDate = scope?.startDate ?? calendarDate(month.from);
  const endDate = scope?.endDate ?? calendarDate(month.to);
  const appGroupId = scope?.appGroupId;
  return useQuery({
    queryKey: ["application-usage", dimension, userId, ids, startDate, endDate, includeOverall, appGroupId],
    queryFn:
      accessToken && ids.length
        ? ({ signal }) => {
            const options = { accessToken, startDate, endDate, signal, appGroupId };
            return customerAppsUsageCall(ids, includeOverall, options, dimension);
          }
        : skipToken,
  });
}
