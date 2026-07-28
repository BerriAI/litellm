import { useSearchParams } from "next/navigation";
import { useCallback } from "react";

import { navigateWithParams } from "@/app/(dashboard)/navigateWithParams";

export const LOG_ID_QUERY_PARAM = "log_id";
export const SESSION_ID_QUERY_PARAM = "session_id";

export interface LogDetailRouting {
  logId: string | null;
  sessionId: string | null;
  openLog: (requestId: string) => void;
  openSession: (sessionId: string, requestId: string | null) => void;
  selectLog: (requestId: string) => void;
  close: () => void;
}

export function useLogDetailRouting(): LogDetailRouting {
  const searchParams = useSearchParams();

  const openLog = useCallback((requestId: string) => {
    navigateWithParams((params) => {
      params.set(LOG_ID_QUERY_PARAM, requestId);
      params.delete(SESSION_ID_QUERY_PARAM);
    });
  }, []);

  const openSession = useCallback((sessionId: string, requestId: string | null) => {
    navigateWithParams((params) => {
      params.set(SESSION_ID_QUERY_PARAM, sessionId);
      if (requestId === null) {
        params.delete(LOG_ID_QUERY_PARAM);
      } else {
        params.set(LOG_ID_QUERY_PARAM, requestId);
      }
    });
  }, []);

  const selectLog = useCallback((requestId: string) => {
    navigateWithParams((params) => {
      params.set(LOG_ID_QUERY_PARAM, requestId);
    }, "replace");
  }, []);

  const close = useCallback(() => {
    navigateWithParams((params) => {
      params.delete(LOG_ID_QUERY_PARAM);
      params.delete(SESSION_ID_QUERY_PARAM);
    });
  }, []);

  return {
    logId: searchParams?.get(LOG_ID_QUERY_PARAM) ?? null,
    sessionId: searchParams?.get(SESSION_ID_QUERY_PARAM) ?? null,
    openLog,
    openSession,
    selectLog,
    close,
  };
}
