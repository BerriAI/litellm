import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";

import researchTrace from "../__fixtures__/research_trace.json";
import swarmTrace from "../__fixtures__/swarm_trace.json";
import type { TracesApi } from "../api";
import { initialRunSelection } from "../detail/run/useRunTree";
import { traceDetailQuery, traceKeys } from "../queries";
import type { TraceRef } from "../routing";
import type { SpanDetail, Trace } from "../types";
import { createTracePrefetcher, prefetchOrder } from "./tracePrefetch";

vi.mock("../../../networking", () => ({ getProxyBaseUrl: () => "http://proxy.test" }));

const research = researchTrace as Trace;
const swarm = swarmTrace as Trace;
const TOKEN = "sk-test";

const refs = (count: number): TraceRef[] => Array.from({ length: count }, (_, i) => ({ traceId: `t${i}` }));

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
    await prefetcherFor(api).prefetcher.warm(refs(10));
    expect(flight.max).toBe(4);
    expect(trace).toHaveBeenCalledTimes(10);
  });

  it("skips runs that already have data or a read in flight", async () => {
    const { api, trace } = fakeTraces(async () => research);
    const { queryClient, prefetcher } = prefetcherFor(api);
    queryClient.setQueryData(traceKeys.trace(TOKEN, { traceId: "t0" }), { pages: [research], pageParams: [null] });
    void queryClient.fetchInfiniteQuery(traceDetailQuery(api, TOKEN, { traceId: "t1" }));
    expect(trace).toHaveBeenCalledTimes(1);
    await prefetcher.warm(refs(3));
    expect(trace.mock.calls.map(([id]) => id)).toEqual(["t1", "t2"]);
  });

  it("drops queued runs when a new set of runs replaces them", async () => {
    const first = deferred<Trace>();
    const { api, trace } = fakeTraces((traceId) => (traceId === "t0" ? first.promise : Promise.resolve(research)));
    const { prefetcher } = prefetcherFor(api, new QueryClient(), 1);
    const stale = prefetcher.warm(refs(3));
    await flush();
    const fresh = prefetcher.warm([{ traceId: "t9" }]);
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
    await prefetcher.warm([{ traceId: "t0", traceRef: "ref0" }]);
    const { selectedId } = initialRunSelection(fixture);
    expect(span.mock.calls).toEqual([["t0", selectedId, "ref0"]]);
    expect(queryClient.getQueryData(traceKeys.span(TOKEN, "t0", "ref0", selectedId))).toEqual({ span_id: selectedId });
  });

  it("reads no span when the first page has no root to select", async () => {
    const { api, span } = fakeTraces(async () => ({ ...research, spans: [] }));
    await prefetcherFor(api).prefetcher.warm([{ traceId: "t0" }]);
    expect(span).not.toHaveBeenCalled();
  });

  it("leaves no cached error behind for a failed read and does not read that run again", async () => {
    const { api, trace } = fakeTraces(async () => {
      throw new Error("too large");
    });
    const { queryClient, prefetcher } = prefetcherFor(api);
    await prefetcher.warm([{ traceId: "t0" }]);
    expect(queryClient.getQueryCache().find({ queryKey: traceKeys.trace(TOKEN, { traceId: "t0" }) })).toBeUndefined();
    await prefetcher.warm([{ traceId: "t0" }]);
    expect(trace).toHaveBeenCalledTimes(1);
  });
});

describe("prefetchOrder", () => {
  const runs = refs(6);

  it("puts the next then previous run first and never includes the open run", () => {
    const order = prefetchOrder(runs, runs.slice(0, 4), runs[2]);
    expect(order.map((ref) => ref.traceId)).toEqual(["t3", "t1", "t0"]);
  });

  it("is just the visible rows when no run is open", () => {
    expect(prefetchOrder(runs, runs.slice(1, 3), null).map((ref) => ref.traceId)).toEqual(["t1", "t2"]);
  });

  it("treats an empty trace ref like a missing one", () => {
    const open = { traceId: "t1", traceRef: "" };
    expect(prefetchOrder(runs, runs.slice(0, 3), open).map((ref) => ref.traceId)).toEqual(["t2", "t0"]);
  });
});
