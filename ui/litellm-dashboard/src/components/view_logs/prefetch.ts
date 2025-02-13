import { QueryClient } from "@tanstack/react-query";
import { uiSpendLogDetailsCall } from "../networking";
import { LogEntry } from "./columns";

export const prefetchLogDetails = (
  logs: LogEntry[], 
  formattedStartTime: string,
  accessToken: string,
  queryClient: QueryClient
) => {
  logs.forEach((log) => {
    if (log.request_id) {
      queryClient.prefetchQuery({
        queryKey: ["logDetails", log.request_id, formattedStartTime],
        queryFn: () => uiSpendLogDetailsCall(accessToken, log.request_id, formattedStartTime),
        staleTime: 10 * 60 * 1000, // 10 minutes
        gcTime: 10 * 60 * 1000, // 10 minutes
      }).catch((error) => {
        console.error(`Failed to prefetch details for log: ${log.request_id}`, error);
      });
    }
  });
};