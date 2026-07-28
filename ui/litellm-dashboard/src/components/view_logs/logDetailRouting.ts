import { useSearchParams } from "next/navigation";
import { useCallback } from "react";

import { navigateWithParams } from "@/app/(dashboard)/navigateWithParams";

export const LOG_ID_QUERY_PARAM = "log_id";

export interface LogDetailRouting {
  logId: string | null;
  openLog: (requestId: string) => void;
  close: () => void;
}

export function useLogDetailRouting(): LogDetailRouting {
  const searchParams = useSearchParams();

  const openLog = useCallback((requestId: string) => {
    navigateWithParams((params) => {
      params.set(LOG_ID_QUERY_PARAM, requestId);
    });
  }, []);

  const close = useCallback(() => {
    navigateWithParams((params) => {
      params.delete(LOG_ID_QUERY_PARAM);
    });
  }, []);

  return {
    logId: searchParams?.get(LOG_ID_QUERY_PARAM) ?? null,
    openLog,
    close,
  };
}
