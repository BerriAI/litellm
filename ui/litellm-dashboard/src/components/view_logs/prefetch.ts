import { QueryClient } from "@tanstack/react-query";
import { uiSpendLogDetailsCall } from "../networking";
import { LogEntry } from "./columns";

export interface PrefetchedLog {
  id: string;
  messages: any;
  response: any;
}

export const prefetchLogDetails = async (
  logs: LogEntry[],
  formattedStartTime: string,
  accessToken: string,
  queryClient: QueryClient,
) => {
  console.log("prefetchLogDetails called with", logs.length, "logs");

  const promises = logs.map((log) => {
    if (log.request_id) {
      console.log("Prefetching details for request_id:", log.request_id);
      return queryClient.prefetchQuery({
        queryKey: ["logDetails", log.request_id, formattedStartTime],
        queryFn: async () => {
          console.log("Fetching details for", log.request_id);
          const result = (await uiSpendLogDetailsCall(
            accessToken,
            log.request_id,
            formattedStartTime,
          )) as PrefetchedLog;
          console.log("Received details for", log.request_id, ":", result ? "success" : "failed");
          return result;
        },
        staleTime: 10 * 60 * 1000, // 10 minutes
        gcTime: 10 * 60 * 1000, // 10 minutes
      });
    }
  });

  try {
    const results = await Promise.all(promises);
    console.log("All prefetch promises completed:", results.length);
    return results;
  } catch (error) {
    console.error("Error in prefetchLogDetails:", error);
    throw error;
  }
};
