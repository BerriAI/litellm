import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import { TracesApiContext, type TraceListRequest, type TracesApi, type TraceWindow } from "../api";
import type { TracePage } from "../types";
import { useAgentTraces, useTraceAgents } from "./useAgentTraces";

const RANGE = { hours: 24, anchorMs: Date.parse("2026-10-01T00:00:00Z") };

const fakeApi = (
  list: (request: TraceListRequest) => Promise<TracePage>,
  agents: (window: TraceWindow) => Promise<readonly string[]> = async () => [],
) => {
  const spies = { list: vi.fn(list), agents: vi.fn(agents) };
  return { spies, api: { live: true, ...spies } as unknown as TracesApi };
};

const wrapperFor = (api: TracesApi) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return function Providers({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>
        <TracesApiContext.Provider value={api}>{children}</TracesApiContext.Provider>
      </QueryClientProvider>
    );
  };
};

describe("useAgentTraces", () => {
  it("asks the server for the picked agent on every page and refetches when the agent changes", async () => {
    const { api, spies } = fakeApi(async ({ cursor }) => ({ data: [], next_cursor: cursor ? null : "next" }));
    const base = { accessToken: "sk", range: RANGE, enabled: true };
    const { result, rerender } = renderHook(({ agent }) => useAgentTraces({ ...base, agent }), {
      wrapper: wrapperFor(api),
      initialProps: { agent: "claude-code" },
    });
    await waitFor(() => expect(result.current.hasMore).toBe(true));
    act(() => result.current.loadMore());
    await waitFor(() => expect(spies.list).toHaveBeenCalledTimes(2));
    expect(spies.list.mock.calls.map(([request]) => [request.agent, request.cursor ?? null])).toEqual([
      ["claude-code", null],
      ["claude-code", "next"],
    ]);

    rerender({ agent: "research-agent" });
    await waitFor(() => expect(spies.list).toHaveBeenCalledTimes(3));
    expect(spies.list.mock.lastCall?.[0]).toMatchObject({ agent: "research-agent" });
    expect(spies.list.mock.lastCall?.[0].cursor ?? null).toBeNull();
  });
});

describe("useTraceAgents", () => {
  it("lists every agent in the range's window", async () => {
    const { api, spies } = fakeApi(
      async () => ({ data: [], next_cursor: null }),
      async () => ["claude-code", "research-agent"],
    );
    const { result } = renderHook(() => useTraceAgents({ accessToken: "sk", range: RANGE, enabled: true }), {
      wrapper: wrapperFor(api),
    });
    await waitFor(() => expect(result.current).toEqual(["claude-code", "research-agent"]));
    expect(spies.agents).toHaveBeenCalledWith({ startMs: RANGE.anchorMs - 24 * 3_600_000, endMs: RANGE.anchorMs });
  });

  it("falls back to no extra agents when the list cannot be read", async () => {
    const { api, spies } = fakeApi(
      async () => ({ data: [], next_cursor: null }),
      async () => {
        throw new Error("unavailable");
      },
    );
    const { result } = renderHook(() => useTraceAgents({ accessToken: "sk", range: RANGE, enabled: true }), {
      wrapper: wrapperFor(api),
    });
    await waitFor(() => expect(spies.agents).toHaveBeenCalled());
    expect(result.current).toEqual([]);
  });
});
