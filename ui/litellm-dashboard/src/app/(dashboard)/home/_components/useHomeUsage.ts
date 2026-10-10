import { useEffect, useMemo, useState } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";
import { gatewayDailyActivityCall } from "@/components/networking";
import { toDailyData } from "@/components/UsagePage/dailyActivityApi";
import type { DailyData } from "@/components/UsagePage/types";
import { all_admin_roles } from "@/utils/roles";
import { useAggregatedDailyActivity } from "@/app/(dashboard)/usage/_components/hooks/useAggregatedDailyActivity";
import { ENTITY_API } from "@/app/(dashboard)/usage/_components/components/EntityUsage/entityFetchFns";
import type { GatewayActivity } from "@/app/(dashboard)/usage/_components/components/gatewayActivity";
import { EMPTY_DAILY_ACTIVITY_METADATA, type DailyActivityRequest } from "@/components/UsagePage/dailyActivityApi";
import {
  overviewTotals,
  type OverviewTotals,
} from "@/app/(dashboard)/usage/_components/components/overview/overviewData";

export const HOME_USAGE_DAYS = 7;

export interface HomeUsage {
  results: readonly DailyData[];
  totals: OverviewTotals;
  loading: boolean;
  requestCountsPending: boolean;
  failed: boolean;
  budget: number | null;
}

/** Last HOME_USAGE_DAYS of the signed-in user's view of usage: deployment-wide for admins, their own otherwise. */
export function useHomeUsage(): HomeUsage {
  const { accessToken, userRole, userId } = useAuthorized();
  const { data: currentUser } = useCurrentUser();
  const isAdmin = all_admin_roles.includes(userRole || "");
  const range = useMemo(() => {
    const endTime = new Date();
    const startTime = new Date(endTime.getTime() - HOME_USAGE_DAYS * 24 * 60 * 60 * 1000);
    return { startTime, endTime };
  }, []);

  const request = useMemo<DailyActivityRequest | null>(
    () =>
      accessToken && (isAdmin || userId)
        ? { accessToken, ...range, entityIds: isAdmin ? null : [userId as string] }
        : null,
    [accessToken, isAdmin, range, userId],
  );
  const { data, loading, failed } = useAggregatedDailyActivity({
    fetch: () => ENTITY_API.user.aggregated(request as DailyActivityRequest),
    enabled: request !== null,
    deps: [accessToken, isAdmin, userId, range.startTime.toISOString()],
  });

  const [gateway, setGateway] = useState<GatewayActivity | null>(null);
  useEffect(() => {
    if (!isAdmin || !accessToken) return;
    let cancelled = false;
    gatewayDailyActivityCall(accessToken, range.startTime, range.endTime)
      .then((activity: GatewayActivity) => {
        const recorded = activity.total_successful_requests + activity.total_failed_requests > 0;
        if (!cancelled) setGateway(recorded ? activity : null);
      })
      .catch(() => {
        if (!cancelled) setGateway(null);
      });
    return () => {
      cancelled = true;
    };
  }, [isAdmin, accessToken, range]);

  const results = useMemo(() => toDailyData(data), [data]);
  const totals = useMemo(
    () => overviewTotals(data.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA, gateway),
    [data.metadata, gateway],
  );

  return {
    results,
    totals,
    loading,
    requestCountsPending: loading && gateway === null,
    failed,
    budget: currentUser?.max_budget ?? null,
  };
}
