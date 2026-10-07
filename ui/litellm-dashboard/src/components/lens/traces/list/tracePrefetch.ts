import { type QueryClient, type QueryKey, useQueryClient } from "@tanstack/react-query";
import { uniqBy } from "es-toolkit";
import { Semaphore } from "es-toolkit/promise";
import { useEffect, useEffectEvent, useMemo } from "react";

import { type TracesApi, useTracesApi } from "../api";
import { initialRunSelection } from "../detail/run/useRunTree";
import { spanDetailQuery, traceDetailQuery } from "../queries";
import { type TraceRef, traceKey } from "../routing";

const TRACE_PREFETCH_CONCURRENCY = 4;

interface TracePrefetcherDeps {
  readonly queryClient: QueryClient;
  readonly traces: TracesApi;
  readonly accessToken: string;
  readonly concurrency: number;
}

interface TracePrefetcher {
  warm(refs: readonly TraceRef[]): Promise<void>;
  cancel(): void;
}

export function createTracePrefetcher({
  queryClient,
  traces,
  accessToken,
  concurrency,
}: TracePrefetcherDeps): TracePrefetcher {
  const semaphore = new Semaphore(concurrency);
  const failed = new Set<string>();
  let batch = new AbortController();

  const present = (queryKey: QueryKey): boolean => {
    const state = queryClient.getQueryState(queryKey);
    return state?.data !== undefined || state?.fetchStatus === "fetching";
  };

  // A cached error with no data would make the suspense open throw instead of reading again.
  const forgetUnobserved = (queryKey: QueryKey) => {
    const query = queryClient.getQueryCache().find({ queryKey, exact: true });
    if (query && query.getObserversCount() === 0) queryClient.removeQueries({ queryKey, exact: true });
  };

  const warmSpan = async (ref: TraceRef, spanId: string) => {
    const query = spanDetailQuery(traces, accessToken, { ...ref, spanId });
    if (!spanId || present(query.queryKey)) return;
    await queryClient.fetchQuery(query).catch(() => forgetUnobserved(query.queryKey));
  };

  const warmRun = async (ref: TraceRef) => {
    const query = traceDetailQuery(traces, accessToken, ref);
    if (failed.has(traceKey(ref)) || present(query.queryKey)) return;
    try {
      const data = await queryClient.fetchInfiniteQuery(query);
      await warmSpan(ref, initialRunSelection(data.pages[0]).selectedId);
    } catch {
      failed.add(traceKey(ref));
      forgetUnobserved(query.queryKey);
    }
  };

  const warmQueued = async (ref: TraceRef, signal: AbortSignal) => {
    await semaphore.acquire();
    try {
      if (!signal.aborted) await warmRun(ref);
    } finally {
      semaphore.release();
    }
  };

  const cancel = () => batch.abort();

  const warm = async (refs: readonly TraceRef[]) => {
    cancel();
    batch = new AbortController();
    const { signal } = batch;
    await Promise.all(refs.map((ref) => warmQueued(ref, signal)));
  };

  return { warm, cancel };
}

/** J / K neighbours of the open run first (next, then previous), then the rows on screen; never the open run. */
export function prefetchOrder(
  runs: readonly TraceRef[],
  visible: readonly TraceRef[],
  open: TraceRef | null,
): TraceRef[] {
  const openKey = open ? traceKey(open) : null;
  const index = openKey === null ? -1 : runs.findIndex((run) => traceKey(run) === openKey);
  const neighbours = index < 0 ? [] : [runs[index + 1], runs[index - 1]].filter((run) => run !== undefined);
  return uniqBy([...neighbours, ...visible], traceKey).filter((ref) => traceKey(ref) !== openKey);
}

interface TracePrefetchOptions {
  readonly runs: readonly TraceRef[];
  readonly visible: readonly TraceRef[];
  readonly open: TraceRef | null;
  readonly paused: boolean;
}

export function useTracePrefetch(
  accessToken: string,
  active: boolean,
  { runs, visible, open, paused }: TracePrefetchOptions,
) {
  const queryClient = useQueryClient();
  const traces = useTracesApi(accessToken);
  const prefetcher = useMemo(() => {
    const deps = { queryClient, traces, accessToken, concurrency: TRACE_PREFETCH_CONCURRENCY };
    return createTracePrefetcher(deps);
  }, [queryClient, traces, accessToken]);
  const refs = active && !paused ? prefetchOrder(runs, visible, open) : [];
  const refsKey = refs.map(traceKey).join("\n");
  const warm = useEffectEvent(() => void prefetcher.warm(refs));
  useEffect(() => {
    warm();
    return prefetcher.cancel;
  }, [prefetcher, refsKey]);
}
