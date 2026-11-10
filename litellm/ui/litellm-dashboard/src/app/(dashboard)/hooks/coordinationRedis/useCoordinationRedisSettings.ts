import { useMutation, UseMutationResult, useQuery, useQueryClient, UseQueryResult } from "@tanstack/react-query";
import {
  getCoordinationRedisSettingsCall,
  testCoordinationRedisConnectionCall,
  updateCoordinationRedisSettingsCall,
} from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import type {
  CoordinationRedisSettings,
  CoordinationRedisSettingsResponse,
  CoordinationRedisTestResponse,
} from "@/app/(dashboard)/caching/_components/coordination_redis_settings/types";
import { createQueryKeys } from "../common/queryKeysFactory";

export const coordinationRedisKeys = createQueryKeys("coordinationRedis");

export const useCoordinationRedisSettings = (): UseQueryResult<CoordinationRedisSettingsResponse> => {
  const { accessToken } = useAuthorized();
  return useQuery<CoordinationRedisSettingsResponse>({
    queryKey: coordinationRedisKeys.list({}),
    queryFn: async () => getCoordinationRedisSettingsCall(accessToken!),
    enabled: Boolean(accessToken),
  });
};

export const useUpdateCoordinationRedisSettings = (): UseMutationResult<void, Error, CoordinationRedisSettings> => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<void, Error, CoordinationRedisSettings>({
    mutationFn: async (settings) => updateCoordinationRedisSettingsCall(accessToken!, settings),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: coordinationRedisKeys.all }),
  });
};

export const useTestCoordinationRedisConnection = (): UseMutationResult<
  CoordinationRedisTestResponse,
  Error,
  CoordinationRedisSettings
> => {
  const { accessToken } = useAuthorized();

  return useMutation<CoordinationRedisTestResponse, Error, CoordinationRedisSettings>({
    mutationFn: async (settings) => testCoordinationRedisConnectionCall(accessToken!, settings),
  });
};
