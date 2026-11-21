import { skipToken, useQuery } from "@tanstack/react-query";
import useAuthorized from "../../hooks/useAuthorized";
import { customerAppsUsageCall } from "../../customers/_components/customerUsageApi";
import { calendarDate, customerPresetRange } from "../../customers/_components/customerUsageRange";

export function useApplicationUsage(
  ids: readonly string[],
  includeOverall: boolean,
  dimension: "appId" | "appGroupId" = "appId",
) {
  const { accessToken, userId } = useAuthorized();
  const month = customerPresetRange("month");
  const startDate = calendarDate(month.from);
  const endDate = calendarDate(month.to);
  return useQuery({
    queryKey: ["application-usage", dimension, userId, ids, startDate, endDate, includeOverall],
    queryFn:
      accessToken && ids.length
        ? ({ signal }) => {
            const options = { accessToken, startDate, endDate, signal };
            return customerAppsUsageCall(ids, includeOverall, options, dimension);
          }
        : skipToken,
  });
}
