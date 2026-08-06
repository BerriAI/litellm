import { parseAsString, useQueryStates } from "nuqs";
import { useCallback } from "react";

import { useDetailHistoryClose } from "@/app/(dashboard)/hooks/useDetailHistoryClose";

export const LOG_ID_QUERY_PARAM = "log_id";
export const SESSION_ID_QUERY_PARAM = "session_id";

export interface LogDetailRouting {
  logId: string | null;
  sessionId: string | null;
  openLog: (requestId: string) => void;
  openSession: (sessionId: string, requestId: string | null) => void;
  selectLog: (requestId: string, sessionId?: string | null) => void;
  close: () => void;
}

export function useLogDetailRouting(): LogDetailRouting {
  const [{ log_id, session_id }, setParams] = useQueryStates(
    { log_id: parseAsString, session_id: parseAsString },
    { history: "push" },
  );

  const clearParams = useCallback(() => {
    void setParams({ log_id: null, session_id: null }, { history: "replace" });
  }, [setParams]);
  const { markOpened, close } = useDetailHistoryClose(clearParams);

  const openLog = useCallback(
    (requestId: string) => {
      markOpened();
      void setParams({ log_id: requestId, session_id: null });
    },
    [markOpened, setParams],
  );

  const openSession = useCallback(
    (sessionId: string, requestId: string | null) => {
      markOpened();
      void setParams({ session_id: sessionId, log_id: requestId });
    },
    [markOpened, setParams],
  );

  const selectLog = useCallback(
    (requestId: string, sessionId?: string | null) => {
      void setParams(sessionId ? { log_id: requestId, session_id: sessionId } : { log_id: requestId }, {
        history: "replace",
      });
    },
    [setParams],
  );

  return {
    logId: log_id,
    sessionId: session_id,
    openLog,
    openSession,
    selectLog,
    close,
  };
}
