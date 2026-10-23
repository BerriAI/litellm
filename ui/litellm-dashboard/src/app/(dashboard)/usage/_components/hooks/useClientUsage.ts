import { useCallback, useEffect, useState } from "react";
import { uiSpendLogsCall } from "@/components/networking";
import {
  CLIENT_REQUEST_LIMIT,
  clientUsageFromLogs,
  emptyClientUsage,
  type ClientUsageResult,
  type UsageClient,
} from "./clientUsageData";

interface ClientUsageOptions {
  accessToken: string | null;
  startTime: Date | null;
  endTime: Date | null;
  userId: string | null;
  teamId?: string | null;
  client: UsageClient | null;
}

const utcSecond = (time: number) => new Date(time).toISOString().slice(0, 19).replace("T", " ");

export function useClientUsage({ accessToken, startTime, endTime, userId, teamId = null, client }: ClientUsageOptions) {
  const start = startTime?.getTime() ?? Number.NaN;
  const end = endTime?.getTime() ?? Number.NaN;
  const scopeError = (() => {
    if (client === null) return null;
    if (!userId && !teamId) return "Select a user or team to view client usage";
    if (!Number.isFinite(start) || !Number.isFinite(end) || start > end) return "Select a valid date range";
    if (!accessToken) return "Sign in to view client usage";
    return null;
  })();
  const enabled = client !== null && scopeError === null;
  const [attempt, setAttempt] = useState(0);
  const reload = useCallback(() => setAttempt((current) => current + 1), []);
  const key = JSON.stringify([attempt, accessToken, start, end, userId, teamId, client]);
  const [settled, setSettled] = useState<{ key: string; result: ClientUsageResult } | null>(null);

  useEffect(() => {
    if (!enabled) return;
    if (!accessToken || !client) return;
    if (!userId && !teamId) return;
    let cancelled = false;
    const scope = { userId, teamId, client, startTime: new Date(start), endTime: new Date(end) };
    const request = {
      accessToken,
      start_date: utcSecond(Math.floor(start / 1000) * 1000),
      end_date: utcSecond(Math.ceil(end / 1000) * 1000),
      page: 1,
      page_size: CLIENT_REQUEST_LIMIT,
      params: {
        ...(userId ? { user_id: userId } : {}),
        ...(teamId ? { team_id: teamId } : {}),
        sort_by: "startTime",
        sort_order: "desc" as const,
        group_by_session: false,
      },
    };
    uiSpendLogsCall(request)
      .then((response: unknown) => {
        if (!cancelled) setSettled({ key, result: clientUsageFromLogs(response, scope) });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setSettled({
            key,
            result: emptyClientUsage(error instanceof Error ? error.message : "Could not load client usage"),
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, accessToken, start, end, userId, teamId, client, key]);

  const current = enabled && settled?.key === key ? settled.result : emptyClientUsage(scopeError);
  return { ...current, loading: enabled && settled?.key !== key, reload };
}
