"use client";

import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import moment from "moment";

import { uiSpendLogsCall } from "../../../networking";
import type { LogEntry } from "../../../logs/types";

/** Spend-log timestamps are written when the call finishes, so pad the span start on both sides. */
const LOOKUP_PAD_MINUTES = 30;
const SPEND_LOG_TIME_FORMAT = "YYYY-MM-DD HH:mm:ss";

export const spanLogWindow = (spanStartMs: number): { start_date: string; end_date: string } => ({
  start_date: moment.utc(spanStartMs).subtract(LOOKUP_PAD_MINUTES, "minutes").format(SPEND_LOG_TIME_FORMAT),
  end_date: moment.utc(spanStartMs).add(LOOKUP_PAD_MINUTES, "minutes").format(SPEND_LOG_TIME_FORMAT),
});

/**
 * The LiteLLM request log behind an LLM span, looked up around the span's own time rather than
 * the Request Logs tab's window, so runs older than that window still resolve.
 */
export function useSpanRequestLog(
  accessToken: string,
  requestId: string | null,
  spanStartMs: number,
  enabled: boolean,
) {
  const fetchLog = async (): Promise<LogEntry | null> => {
    if (requestId === null) return null;
    const logsOptions: Parameters<typeof uiSpendLogsCall>[0] = {
      accessToken,
      ...spanLogWindow(spanStartMs),
      page: 1,
      page_size: 1,
      params: { request_id: requestId },
    };
    const response = await uiSpendLogsCall(logsOptions);
    return response.data.find((log: LogEntry) => log.request_id === requestId) ?? null;
  };
  const queryOptions: UseQueryOptions<LogEntry | null> = {
    queryKey: ["logs", "spanRequest", requestId, spanStartMs, accessToken],
    queryFn: fetchLog,
    enabled: enabled && requestId !== null,
    staleTime: Infinity,
  };
  return useQuery(queryOptions);
}
