import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/http/client";

import researchTrace from "../__fixtures__/research_trace.json";
import swarmTrace from "../__fixtures__/swarm_trace.json";
import type { TracesApi } from "../api";
import { initialRunSelection } from "../detail/run/useRunTree";
import { TRACE_DETAIL_STALE_MS, traceKeys } from "../queries";
import type { TraceRef } from "../routing";
import type { SpanDetail, Trace, TraceSummary } from "../types";
import {
  createTracePrefetcher,
  prefetchPlan,
  type PrefetchTarget,
  TRACE_PREFETCH_ENDED_AFTER_MS,
  TRACE_PREFETCH_GC_MS,
  TRACE_PREFETCH_INTENT_MS,
  TRACE_PREFETCH_VISIBLE_ROWS,
} from "./tracePrefetch";

vi.mock("../../../networking", () => ({ getProxyBaseUrl: () => "http://proxy.test" }));

const research = researchTrace as Trace;
const swarm = swarmTrace as Trace;
const TOKEN = "sk-test";

const refs = (count: number): TraceRef[] => Array.from({ length: count }, (_, i) => ({ traceId: `t${i}` }));
const rows = (count: number, span = false): PrefetchTarget[] => refs(count).map((ref) => ({ ref, span }));

interface Deferred<T> {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
  readonly reject: (reason: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  const handlers: { resolve?: (value: T) => void; reject?: (reason: unknown) => void } = {};
  const promise = new Promise<T>((resolve, reject) => Object.assign(handlers, { resolve, reject }));
  return { promise, resolve: (value) => handlers.resolve?.(value), reject: (reason) => handlers.reject?.(reason) };
}

function fakeTraces(read: (traceId: string) => Promise<Trace>) {
  const trace = vi.fn((traceId: string) => read(traceId));
  const span = vi.fn(
    async (traceId: string, spanId: string): Promise<SpanDetail> => ({ span_id: spanId }) as SpanDetail,
  );
  const api = { live: true, trace, span } as unknown as TracesApi;
  return { api, trace, span };
}

const prefetcherFor = (traces: TracesApi, queryClient = new QueryClient(), concurrency = 4) => {
  const deps = { queryClient, traces, accessToken: TOKEN, concurrency };
  return { queryClient, prefetcher: createTracePrefetcher(deps) };
};

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("createTracePrefetcher", () => {
  it("never has more trace reads in flight than the concurrency cap, and reads every run", async () => {
    const flight = { now: 0, max: 0 };
    const { api, trace } = fakeTraces(async () => {
      flight.now += 1;
      flight.max = Math.max(flight.max, flight.now);
      await flush();
      flight.now -= 1;
      return { ...research, spans: [] };
    });
    await prefetcherFor(api).prefetcher.warm(rows(10));
    expect(flight.max).toBe(4);
    expect(trace).toHaveBeenCalledTimes(10);
  });

  it("keeps a cached run even after it goes stale, and reads only the uncached ones", async () => {
    const { api, trace } = fakeTraces(async () => research);
    const { queryClient, prefetcher } = prefetcherFor(api);
    const staleAt = Date.now() - TRACE_DETAIL_STALE_MS - 1;
    const page = { pages: [research], pageParams: [null] };
    queryClient.setQueryData(traceKeys.trace(TOKEN, { traceId: "t0" }), page, { updatedAt: staleAt });
    await prefetcher.warm(rows(2));
    expect(trace.mock.calls.map(([id]) => id)).toEqual(["t1"]);
  });

  it("does not read an on-screen run again after its prefetched copy is dropped", async () => {
    const { api, trace } = fakeTraces(async () => research);
    const { queryClient, prefetcher } = prefetcherFor(api);
    await prefetcher.warm(rows(2));
    queryClient.clear();
    await prefetcher.warm(rows(3));
    expect(trace.mock.calls.map(([id]) => id)).toEqual(["t0", "t1", "t2"]);
  });

  it("drops queued runs when a new set of runs replaces them", async () => {
    const first = deferred<Trace>();
    const { api, trace } = fakeTraces((traceId) => (traceId === "t0" ? first.promise : Promise.resolve(research)));
    const { prefetcher } = prefetcherFor(api, new QueryClient(), 1);
    const stale = prefetcher.warm(rows(3));
    await flush();
    const fresh = prefetcher.warm([{ ref: { traceId: "t9" }, span: false }]);
    first.resolve(research);
    await Promise.all([stale, fresh]);
    expect(trace.mock.calls.map(([id]) => id)).toEqual(["t0", "t9"]);
  });

  it.each([
    ["the root agent of a healthy run", research],
    ["the first visible failure of a failed run", swarm],
  ])("reads the span a fresh open selects: %s", async (_label, fixture) => {
    const { api, span } = fakeTraces(async () => fixture);
    const { queryClient, prefetcher } = prefetcherFor(api);
    await prefetcher.warm([{ ref: { traceId: fixture.summary.trace_id }, span: true }]);
    const { selectedId } = initialRunSelection(fixture);
    const { trace_id: traceId, trace_ref: traceRef } = fixture.summary;
    expect(span.mock.calls).toEqual([[traceId, selectedId, traceRef]]);
    expect(queryClient.getQueryData(traceKeys.span(TOKEN, traceId, traceRef, selectedId))).toEqual({
      span_id: selectedId,
    });
  });

  it("keys the span by the trace ref the detail resolved, where the open run reads it", async () => {
    const resolved = { ...research, summary: { ...research.summary, trace_id: "t0", trace_ref: "resolved" } };
    const { api, span } = fakeTraces(async () => resolved);
    const { queryClient, prefetcher } = prefetcherFor(api);
    await prefetcher.warm([{ ref: { traceId: "t0" }, span: true }]);
    const { selectedId } = initialRunSelection(resolved);
    expect(span.mock.calls).toEqual([["t0", selectedId, "resolved"]]);
    expect(queryClient.getQueryData(traceKeys.span(TOKEN, "t0", "resolved", selectedId))).toBeDefined();
  });

  it("reads no span for a row that is only on screen", async () => {
    const { api, span } = fakeTraces(async () => research);
    await prefetcherFor(api).prefetcher.warm(rows(3));
    expect(span).not.toHaveBeenCalled();
  });

  it("reads the span of an already warmed row once it becomes a neighbour", async () => {
    const { api, trace, span } = fakeTraces(async () => research);
    const { prefetcher } = prefetcherFor(api);
    await prefetcher.warm(rows(1));
    await prefetcher.warm(rows(1, true));
    expect(trace).toHaveBeenCalledOnce();
    expect(span).toHaveBeenCalledOnce();
  });

  it("reads no span when the first page has no root to select", async () => {
    const { api, span } = fakeTraces(async () => ({ ...research, spans: [] }));
    await prefetcherFor(api).prefetcher.warm(rows(1, true));
    expect(span).not.toHaveBeenCalled();
  });

  it("leaves no cached error behind for a failed read and does not read that run again", async () => {
    vi.useFakeTimers();
    const { api, trace } = fakeTraces(async () => {
      throw new Error("too large");
    });
    const { queryClient, prefetcher } = prefetcherFor(api);
    await prefetcher.warm(rows(1, true));
    expect(queryClient.getQueryCache().find({ queryKey: traceKeys.trace(TOKEN, { traceId: "t0" }) })).toBeUndefined();
    await prefetcher.warm(rows(1, true));
    const hovered = prefetcher.intend({ ref: { traceId: "t0" }, span: true });
    await vi.advanceTimersByTimeAsync(TRACE_PREFETCH_INTENT_MS);
    await hovered;
    vi.useRealTimers();
    expect(trace).toHaveBeenCalledTimes(1);
  });

  it("lets go of a warmed run and its step within a minute when nothing opens it", async () => {
    vi.useFakeTimers();
    const { api } = fakeTraces(async () => research);
    const { queryClient, prefetcher } = prefetcherFor(api);
    await prefetcher.warm(rows(1, true));
    expect(queryClient.getQueryCache().getAll()).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(TRACE_PREFETCH_GC_MS);
    vi.useRealTimers();
    expect(queryClient.getQueryCache().getAll()).toHaveLength(0);
  });

  it("does not retry a read that failed during an outage", async () => {
    const { api, trace } = fakeTraces(async () => {
      throw new ApiError("unavailable", 503, { detail: { code: "unavailable" } }, 0);
    });
    await prefetcherFor(api).prefetcher.warm(rows(1));
    expect(trace).toHaveBeenCalledTimes(1);
  });

  it("warms the hovered run and its default span once the pointer rests on it", async () => {
    vi.useFakeTimers();
    const { api, trace, span } = fakeTraces(async () => research);
    const { prefetcher } = prefetcherFor(api);
    const hovered = prefetcher.intend({ ref: { traceId: "t0" }, span: true });
    await vi.advanceTimersByTimeAsync(TRACE_PREFETCH_INTENT_MS);
    await hovered;
    vi.useRealTimers();
    expect(trace).toHaveBeenCalledOnce();
    expect(span).toHaveBeenCalledOnce();
  });

  it("reads nothing for a row the pointer only passes over", async () => {
    vi.useFakeTimers();
    const { api, trace } = fakeTraces(async () => research);
    const { prefetcher } = prefetcherFor(api);
    const passed = prefetcher.intend({ ref: { traceId: "t0" }, span: true });
    await vi.advanceTimersByTimeAsync(TRACE_PREFETCH_INTENT_MS / 2);
    const rested = prefetcher.intend({ ref: { traceId: "t1" }, span: true });
    await vi.advanceTimersByTimeAsync(TRACE_PREFETCH_INTENT_MS);
    await Promise.all([passed, rested]);
    vi.useRealTimers();
    expect(trace.mock.calls.map(([id]) => id)).toEqual(["t1"]);
  });
});

const NOW = Date.parse("2026-10-01T12:00:00Z");
const ENDED = TRACE_PREFETCH_ENDED_AFTER_MS;

const summary = (id: string, endedAgoMs: number, overrides: Partial<TraceSummary> = {}): TraceSummary => ({
  ...research.summary,
  trace_id: id,
  trace_ref: undefined,
  start_time: new Date(NOW - endedAgoMs - 1_000).toISOString(),
  duration_ms: 1_000,
  ...overrides,
});

const ended = (count: number) => Array.from({ length: count }, (_, i) => summary(`t${i}`, ENDED));
const plan = (runs: TraceSummary[], visible: TraceRef[], open: TraceRef | null) =>
  prefetchPlan(runs, visible, open, NOW).map(({ ref, span }) => [ref.traceId, span]);

describe("prefetchPlan", () => {
  const runs = ended(6);

  it("puts the next then previous run first with their spans, and never includes the open run", () => {
    expect(plan(runs, refs(4), { traceId: "t2" })).toEqual([
      ["t3", true],
      ["t1", true],
      ["t0", false],
    ]);
  });

  it("is just the visible rows, without spans, when no run is open", () => {
    expect(plan(runs, refs(3).slice(1), null)).toEqual([
      ["t1", false],
      ["t2", false],
    ]);
  });

  it("treats an empty trace ref like a missing one", () => {
    expect(plan(runs, refs(3), { traceId: "t1", traceRef: "" })).toEqual([
      ["t2", true],
      ["t0", true],
    ]);
  });

  it("caps the on-screen rows it warms but always keeps the neighbours", () => {
    const order = plan(ended(40), refs(40), { traceId: "t30" });
    expect(order.slice(0, 2)).toEqual([
      ["t31", true],
      ["t29", true],
    ]);
    expect(order.slice(2)).toEqual(refs(TRACE_PREFETCH_VISIBLE_ROWS).map((ref) => [ref.traceId, false]));
  });

  it("skips runs that may still be in progress, whether on screen or a neighbour", () => {
    const live = [summary("t0", 0), summary("t1", ENDED), summary("t2", ENDED - 1), summary("t3", ENDED)];
    expect(plan(live, refs(4), { traceId: "t1" })).toEqual([["t3", false]]);
  });

  it("warms a resumable Claude Code session's trace but never its span", () => {
    const session = [summary("t0", ENDED), summary("t1", ENDED, { frameworks: ["claude-code"] })];
    expect(plan(session, [], { traceId: "t0" })).toEqual([["t1", false]]);
  });

  it("ignores rows reported on screen that are no longer in the list", () => {
    expect(plan(runs.slice(0, 2), refs(4), null)).toEqual([
      ["t0", false],
      ["t1", false],
    ]);
  });
});
