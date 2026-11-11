import { useQuery } from "@tanstack/react-query";
import { useCurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";
import { backgroundFetchClient } from "@/lib/http/api";
import { getProxyBaseUrl } from "@/components/networking";
import { hasCapability } from "@/utils/capabilities";
import { deprecationUsageRange, usedDeprecationAlerts } from "./deprecationAlerts";

const POLL_INTERVAL = 5 * 60 * 1000;

export function useDeprecationAlerts(accessToken: string | null) {
  const { data: currentUser } = useCurrentUser(accessToken);
  const userId = currentUser?.user_id;
  const enabled = Boolean(accessToken && userId && hasCapability(currentUser?.user_role, "viewGlobalSpend"));
  const scope = JSON.stringify([getProxyBaseUrl(), userId]);
  const range = deprecationUsageRange(new Date());
  const options = {
    queryKey: ["deprecationAlerts", scope, range],
    enabled,
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const { data: schedule } = await backgroundFetchClient.GET("/model/deprecations", {
        params: { query: { warn_within_days: 30 } },
        signal,
      });
      if (!schedule) throw new Error("Model deprecation information is unavailable");
      const models = [...(schedule.deprecated ?? []), ...(schedule.imminent ?? [])];
      if (models.length === 0) return [];
      const { data: activity } = await backgroundFetchClient.GET("/user/daily/activity/aggregated", {
        params: { query: { ...range, timezone: 0, api_key_limit: 1 } },
        signal,
      });
      if (!activity) throw new Error("Model usage is unavailable");
      const alerts = usedDeprecationAlerts(models, activity.results ?? []);
      return Promise.all(
        alerts.map(async (alert) => {
          const { data: state } = await backgroundFetchClient.GET("/alerting/model-deprecations/state", {
            params: { query: { model: alert.usageModel, deprecation_date: alert.model.deprecation_date } },
            signal,
          });
          if (!state) throw new Error("Alert acknowledgement state is unavailable");
          alert.state = state;
          return alert;
        }),
      );
    },
    staleTime: POLL_INTERVAL,
    refetchInterval: POLL_INTERVAL,
    retry: false,
  };
  const query = useQuery(options);
  return { ...query, data: enabled ? query.data : undefined, scope, range };
}
