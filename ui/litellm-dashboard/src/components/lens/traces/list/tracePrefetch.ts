import { type QueryClient, type QueryKey, useQueryClient } from "@tanstack/react-query";
import { delay, Semaphore } from "es-toolkit/promise";
import { useEffect, useEffectEvent, useMemo } from "react";

import { type TracesApi, useTracesApi } from "../api";
import { initialRunSelection } from "../detail/run/useRunTree";
import { spanDetailQuery, traceDetailQuery } from "../queries";
import { type TraceRef, traceKey, traceRefOf } from "../routing";
import type { Trace, TraceSummary } from "../types";

const TRACE_PREFETCH_CONCURRENCY = 4;
export const TRACE_PREFETCH_VISIBLE_ROWS = 8;
export const TRACE_PREFETCH_ENDED_AFTER_MS = 5 * 60_000;
export const TRACE_PREFETCH_INTENT_MS = 100;
export const TRACE_PREFETCH_GC_MS = 60_000;
const TRACE_PREFETCH_MEMORY = 500;
const RESUMABLE_FRAMEWORKS: ReadonlySet<string> = new Set(["claude-code", "claude-agent-sdk"]);

export interface PrefetchTarget {
  readonly ref: TraceRef;
  readonly span: boolean;
}

interface TracePrefetcherDeps {
  readonly queryClient: QueryClient;
  readonly traces: TracesApi;
  readonly accessToken: string;
  readonly concurrency: number;
}

interface TracePrefetcher {
  warm(targets: readonly PrefetchTarget[]): Promise<void>;
  intend(target: PrefetchTarget): Promise<void>;
  cancel(): void;
}

export function createTracePrefetcher({
  queryClient,
  traces,
  accessToken,
  concurrency,
}: TracePrefetcherDeps): TracePrefetcher {
  const semaphore = new Semaphore(concurrency);
  const read = new Map<string, boolean>();
  const controllers = { batch: new AbortController(), intent: new AbortController() };

  const remember = (ref: TraceRef, ok: boolean) => {
    if (read.size >= TRACE_PREFETCH_MEMORY) read.clear();
    read.set(traceKey(ref), ok);
  };

  const wanted = ({ ref, span }: PrefetchTarget): boolean => {
    const ok = read.get(traceKey(ref));
    return ok === undefined || (ok && span);
  };

  const forgetUnobserved = (queryKey: QueryKey) => {
    const query = queryClient.getQueryCache().find({ queryKey, exact: true });
    if (query && query.getObserversCount() === 0) queryClient.removeQueries({ queryKey, exact: true });
  };

  const warmSpan = async (page: Trace) => {
    const spanId = initialRunSelection(page).selectedId;
    if (!spanId) return;
    const { trace_id: traceId, trace_ref: traceRef } = page.summary;
    const options = spanDetailQuery(traces, accessToken, { traceId, traceRef, spanId });
    const query = { ...options, retry: false, gcTime: TRACE_PREFETCH_GC_MS };
    await queryClient.fetchQuery(query).catch(() => forgetUnobserved(query.queryKey));
  };

  const warmRun = async ({ ref, span }: PrefetchTarget) => {
    const query = { ...traceDetailQuery(traces, accessToken, ref), retry: false, gcTime: TRACE_PREFETCH_GC_MS };
    try {
      const data = await queryClient.ensureInfiniteQueryData(query);
      remember(ref, true);
      if (span) await warmSpan(data.pages[0]);
    } catch {
      remember(ref, false);
      forgetUnobserved(query.queryKey);
    }
  };

  const warmQueued = async (target: PrefetchTarget, signal: AbortSignal) => {
    await semaphore.acquire();
    try {
      if (!signal.aborted) await warmRun(target);
    } finally {
      semaphore.release();
    }
  };

  const cancel = () => {
    controllers.batch.abort();
    controllers.intent.abort();
  };

  const warm = async (targets: readonly PrefetchTarget[]) => {
    controllers.batch.abort();
    controllers.batch = new AbortController();
    const { signal } = controllers.batch;
    await Promise.all(targets.filter(wanted).map((target) => warmQueued(target, signal)));
  };

  const intend = async (target: PrefetchTarget) => {
    controllers.intent.abort();
    controllers.intent = new AbortController();
    const { signal } = controllers.intent;
    if (!wanted(target)) return;
    const waited = await delay(TRACE_PREFETCH_INTENT_MS, { signal }).then(
      () => true,
      () => false,
    );
    if (waited) await warmQueued(target, signal);
  };

  return { warm, intend, cancel };
}

const runEnded = (run: TraceSummary, nowMs: number): boolean =>
  nowMs - (Date.parse(run.start_time) + run.duration_ms) >= TRACE_PREFETCH_ENDED_AFTER_MS;

const resumable = (run: TraceSummary): boolean =>
  (run.frameworks ?? []).some((framework) => RESUMABLE_FRAMEWORKS.has(framework));

export function prefetchTarget(run: TraceSummary, nowMs: number, span: boolean): PrefetchTarget | null {
  if (!runEnded(run, nowMs)) return null;
  return { ref: traceRefOf(run), span: span && !resumable(run) };
}

export function prefetchPlan(
  runs: readonly TraceSummary[],
  visible: readonly TraceRef[],
  open: TraceRef | null,
  nowMs: number,
): PrefetchTarget[] {
  const runKey = (run: TraceSummary) => traceKey(traceRefOf(run));
  const openKey = open ? traceKey(open) : null;
  const index = openKey === null ? -1 : runs.findIndex((run) => runKey(run) === openKey);
  const adjacent = index < 0 ? [] : [runs[index + 1], runs[index - 1]].filter((run) => run !== undefined);
  const byKey = new Map(runs.map((run) => [runKey(run), run]));
  const taken = new Set([openKey, ...adjacent.map(runKey)]);
  const shown = visible
    .map((ref) => byKey.get(traceKey(ref)))
    .filter((run): run is TraceSummary => run !== undefined && !taken.has(runKey(run)));
  const neighbours = adjacent.map((run) => prefetchTarget(run, nowMs, true));
  const rows = shown.map((run) => prefetchTarget(run, nowMs, false)).filter((target) => target !== null);
  return [...neighbours.filter((target) => target !== null), ...rows.slice(0, TRACE_PREFETCH_VISIBLE_ROWS)];
}

interface TracePrefetchOptions {
  readonly runs: readonly TraceSummary[];
  readonly visible: readonly TraceRef[];
  readonly open: TraceRef | null;
  readonly paused: boolean;
}

export function useTracePrefetch(
  accessToken: string,
  active: boolean,
  { runs, visible, open, paused }: TracePrefetchOptions,
): (run: TraceSummary) => void {
  const queryClient = useQueryClient();
  const traces = useTracesApi(accessToken);
  const prefetcher = useMemo(() => {
    const deps = { queryClient, traces, accessToken, concurrency: TRACE_PREFETCH_CONCURRENCY };
    return createTracePrefetcher(deps);
  }, [queryClient, traces, accessToken]);
  const enabled = active && !paused;
  const visibleKey = visible.map(traceKey).join("\n");
  const openKey = open ? traceKey(open) : null;
  const warm = useEffectEvent(() => void prefetcher.warm(prefetchPlan(runs, visible, open, Date.now())));
  useEffect(() => {
    if (!enabled) return;
    warm();
    return prefetcher.cancel;
  }, [prefetcher, enabled, visibleKey, openKey]);
  return (run) => {
    const target = enabled ? prefetchTarget(run, Date.now(), true) : null;
    if (target) void prefetcher.intend(target);
  };
}
