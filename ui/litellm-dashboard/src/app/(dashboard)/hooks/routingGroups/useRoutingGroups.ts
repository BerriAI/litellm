"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useMutation, useQuery, useQueryClient, UseMutationResult, UseQueryResult } from "@tanstack/react-query";
import { getRouterSettingsCall, setCallbacksCall } from "@/components/networking";
import { createQueryKeys } from "../common/queryKeysFactory";
import type { RoutingGroup } from "@/components/routing_groups/types";

const routingGroupsKeys = createQueryKeys("routingGroups");

interface RoutingGroupsQueryData {
  routingGroups: RoutingGroup[];
  routingStrategy: string | null;
  availableStrategies: string[];
}

const fetchRoutingGroups = async (accessToken: string): Promise<RoutingGroupsQueryData> => {
  const data = await getRouterSettingsCall(accessToken);
  const currentValues = data?.current_values ?? {};
  const fields = Array.isArray(data?.fields) ? data.fields : [];
  const routingStrategyField = fields.find((f: any) => f?.field_name === "routing_strategy");

  return {
    routingGroups: Array.isArray(currentValues.routing_groups) ? currentValues.routing_groups : [],
    routingStrategy: currentValues.routing_strategy ?? null,
    availableStrategies: Array.isArray(routingStrategyField?.options) ? routingStrategyField.options : [],
  };
};

export const useRoutingGroups = (): UseQueryResult<RoutingGroupsQueryData> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<RoutingGroupsQueryData>({
    queryKey: routingGroupsKeys.lists(),
    queryFn: () => fetchRoutingGroups(accessToken!),
    enabled: Boolean(accessToken && userId && userRole),
  });
};

export const useSaveRoutingGroups = (): UseMutationResult<unknown, Error, RoutingGroup[]> => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (routingGroups: RoutingGroup[]) =>
      setCallbacksCall(accessToken!, {
        router_settings: { routing_groups: routingGroups },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: routingGroupsKeys.lists() });
    },
  });
};
