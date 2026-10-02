import { queryOptions, skipToken, useQuery } from "@tanstack/react-query";

import { cacheLeakageKeysCall } from "@/components/networking";
import type { EntityType } from "@/components/EntityUsageExport/types";
import type { DailyActivityRequest } from "@/components/UsagePage/dailyActivityApi";
import { ENTITY_API } from "@/app/(dashboard)/usage/_components/components/EntityUsage/entityFetchFns";

export const DAILY_ACTIVITY_STALE_TIME_MS = 60_000;

export const dailyActivityKeys = {
  all: ["dailyActivity"] as const,
  aggregated: (entity: EntityType, request: DailyActivityRequest | null) =>
    [...dailyActivityKeys.all, entity, "aggregated", request] as const,
  modelTopKeys: (entity: EntityType, request: DailyActivityRequest, model: string, byModelGroup: boolean) =>
    [...dailyActivityKeys.all, entity, "modelTopKeys", request, { model, byModelGroup }] as const,
  cacheLeakageKeys: (request: DailyActivityRequest | null) =>
    [...dailyActivityKeys.all, "user", "cacheLeakageKeys", request] as const,
};

export const modelTopKeysQueryOptions = (
  entity: EntityType,
  request: DailyActivityRequest,
  model: string,
  byModelGroup: boolean,
) =>
  queryOptions({
    queryKey: dailyActivityKeys.modelTopKeys(entity, request, model, byModelGroup),
    queryFn: () => ENTITY_API[entity].modelTopKeys(request, model, byModelGroup),
    staleTime: DAILY_ACTIVITY_STALE_TIME_MS,
  });

export type ModelTopKeysQueryOptions = ReturnType<typeof modelTopKeysQueryOptions>;

export const cacheLeakageKeysQueryOptions = (request: DailyActivityRequest | null) =>
  queryOptions({
    queryKey: dailyActivityKeys.cacheLeakageKeys(request),
    queryFn: request ? () => cacheLeakageKeysCall(request) : skipToken,
    staleTime: DAILY_ACTIVITY_STALE_TIME_MS,
  });

export const useAggregatedDailyActivity = (entity: EntityType, request: DailyActivityRequest | null) =>
  useQuery({
    queryKey: dailyActivityKeys.aggregated(entity, request),
    queryFn: request ? () => ENTITY_API[entity].aggregated(request) : skipToken,
    staleTime: DAILY_ACTIVITY_STALE_TIME_MS,
  });
