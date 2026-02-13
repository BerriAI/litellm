import { useQuery } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { uiSpendLogDetailsCall } from "@/components/networking";

/**
 * Hook to lazy-load log details (messages/response) for a specific log entry.
 * Fetches data on-demand when the drawer is open, instead of prefetching all logs.
 *
 * @param requestId - The request_id of the log entry
 * @param startTime - The formatted start time for the query
 * @param enabled - Whether the query should be enabled (e.g., drawer is open)
 */
export const useLogDetails = (
  requestId: string | undefined,
  startTime: string | undefined,
  enabled: boolean,
) => {
  const { accessToken } = useAuthorized();

  return useQuery({
    queryKey: ["logDetails", requestId, startTime, accessToken],
    queryFn: async () => {
      if (!accessToken || !requestId || !startTime) return null;
      return await uiSpendLogDetailsCall(accessToken, requestId, startTime);
    },
    enabled: enabled && !!accessToken && !!requestId && !!startTime,
    staleTime: 10 * 60 * 1000, // 10 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
  });
};
