import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { backgroundFetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { createQueryKeys } from "../common/queryKeysFactory";

export type ReferenceExchangeRates = components["schemas"]["ReferenceExchangeRates"];

export const REFERENCE_RATES_STALE_MS = 60 * 60 * 1000;

export const referenceRatesKeys = createQueryKeys("referenceRates");

export const fetchReferenceRates = async (signal: AbortSignal): Promise<ReferenceExchangeRates> => {
  const { data } = await backgroundFetchClient.GET("/get/reference_exchange_rates", { signal });
  if (data === undefined) throw new Error("The proxy returned no reference exchange rates");
  return data;
};

export const useReferenceRates = (enabled: boolean) => {
  const options: UseQueryOptions<ReferenceExchangeRates> = {
    queryKey: referenceRatesKeys.all,
    queryFn: ({ signal }) => fetchReferenceRates(signal),
    enabled,
    staleTime: REFERENCE_RATES_STALE_MS,
    refetchInterval: REFERENCE_RATES_STALE_MS,
    retry: false,
  };
  return useQuery(options);
};
