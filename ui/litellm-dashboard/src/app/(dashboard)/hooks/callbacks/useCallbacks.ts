import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { getCallbacksCall, getCallbackConfigsCall } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  AlertingObject,
  AlertData,
  AvailableCallback,
  CallbackConfig,
} from "@/components/Settings/LoggingAndAlerts/LoggingCallbacks/types";

export const callbackKeys = createQueryKeys("callbacks");
const callbackConfigKeys = createQueryKeys("callbackConfigs");

export interface CallbacksData {
  callbacks: AlertingObject[];
  alerts: AlertData[];
  availableCallbacks: Record<string, AvailableCallback>;
}

export const useCallbacks = (): UseQueryResult<CallbacksData> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<CallbacksData>({
    queryKey: callbackKeys.list({}),
    queryFn: async () => {
      const data = await getCallbacksCall(accessToken!, userId!, userRole!);
      return {
        callbacks: data.callbacks || [],
        alerts: data.alerts || [],
        availableCallbacks: data.available_callbacks || {},
      };
    },
    enabled: Boolean(accessToken && userId && userRole),
  });
};

export const useCallbackConfigs = (): UseQueryResult<CallbackConfig[]> => {
  const { accessToken } = useAuthorized();
  return useQuery<CallbackConfig[]>({
    queryKey: callbackConfigKeys.list({}),
    queryFn: async () => {
      const data = await getCallbackConfigsCall(accessToken!);
      return data || [];
    },
    enabled: Boolean(accessToken),
  });
};
