import { keepPreviousData, queryOptions, skipToken, useQuery } from "@tanstack/react-query";
import { formatDate, subscriptionUsageCall } from "@/components/networking";
import type { SubscriptionUsageResponse } from "../components/SubscriptionCost/types";

interface Options {
  accessToken: string | null;
  startTime: Date | null;
  endTime: Date | null;
  enabled: boolean;
}

interface Result {
  data: SubscriptionUsageResponse | null;
  loading: boolean;
  failed: boolean;
  refetch: () => Promise<unknown>;
}

interface UsageRequest {
  accessToken: string;
  startTime: Date;
  endTime: Date;
}

const subscriptionUsageQuery = (request: UsageRequest | null) => {
  const queryKey = [
    "subscriptionUsage",
    request ? formatDate(request.startTime) : null,
    request ? formatDate(request.endTime) : null,
  ];
  const queryFn = request
    ? () => subscriptionUsageCall(request.accessToken, request.startTime, request.endTime)
    : skipToken;
  return queryOptions({ queryKey, queryFn, placeholderData: keepPreviousData });
};

export function useSubscriptionUsage({ accessToken, startTime, endTime, enabled }: Options): Result {
  const range = startTime && endTime ? { startTime, endTime } : null;
  const request = enabled && accessToken && range ? { accessToken, ...range } : null;
  const query = useQuery(subscriptionUsageQuery(request));
  return {
    data: request && query.data ? query.data : null,
    loading: request !== null && (query.isPending || query.isPlaceholderData),
    failed: query.isError,
    refetch: query.refetch,
  };
}
