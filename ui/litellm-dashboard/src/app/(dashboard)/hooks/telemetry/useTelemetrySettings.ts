import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { backgroundFetchClient, fetchClient } from "@/lib/http/api";
import { TELEMETRY_WINDOW_MS } from "@/lib/telemetry/uiEvents";
import type { components } from "@/lib/http/schema";
import { proxyAdminTierRoles } from "@/utils/roles";

export type TelemetrySettings = components["schemas"]["TelemetrySettingsResponse"];
export type TelemetryGroup = components["schemas"]["TelemetryGroup"];
export type TelemetryDestination = components["schemas"]["TelemetryDestination"];
export type TelemetrySettingsUpdate = {
  groups: readonly TelemetryGroup[];
  destination: TelemetryDestination | null;
};

export const telemetrySettingsKeys = createQueryKeys("telemetrySettings");

const fetchTelemetrySettings = async (): Promise<TelemetrySettings> =>
  (await backgroundFetchClient.GET("/telemetry/settings")).data as TelemetrySettings;

export const useTelemetrySettings = () => {
  const { accessToken, userRole } = useAuthorized();
  const options = {
    queryKey: telemetrySettingsKeys.detail("current"),
    queryFn: fetchTelemetrySettings,
    enabled: Boolean(accessToken) && proxyAdminTierRoles.includes(userRole || ""),
    staleTime: TELEMETRY_WINDOW_MS,
  };
  return useQuery<TelemetrySettings>(options);
};

export const useUpdateTelemetrySettings = () => {
  const queryClient = useQueryClient();
  return useMutation<TelemetrySettings, Error, TelemetrySettingsUpdate>({
    mutationFn: async ({ groups, destination }) =>
      (await fetchClient.PUT("/telemetry/settings", { body: { groups: [...groups], destination } }))
        .data as TelemetrySettings,
    onSuccess: (settings) => queryClient.setQueryData(telemetrySettingsKeys.detail("current"), settings),
  });
};
