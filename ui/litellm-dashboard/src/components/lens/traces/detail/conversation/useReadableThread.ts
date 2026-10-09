import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { useTracesApi } from "../../api";
import type { Trace } from "../../types";
import { threadDigest } from "./readable";
import type { ThreadTurn } from "./thread";

export function useReadableThread(trace: Trace, turns: readonly ThreadTurn[], accessToken: string, enabled: boolean) {
  const api = useTracesApi(accessToken);
  const digest = useMemo(() => threadDigest(turns), [turns]);
  const fingerprint = useMemo(() => JSON.stringify(digest), [digest]);
  const { trace_id: traceId, trace_ref: traceRef } = trace.summary;
  const query = {
    queryKey: ["lensReadableThread", traceId, traceRef, accessToken, fingerprint],
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const { runReadableAgent } = await import("./readableAgent");
      const request = { trace, turns: digest, accessToken, api, signal };
      return runReadableAgent(request);
    },
    enabled: enabled && digest.length > 0,
    staleTime: Infinity,
    gcTime: 30 * 60_000,
    retry: false,
  };
  return useQuery(query);
}
