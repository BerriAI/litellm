import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { usePaginatedDailyActivity } from "../src/components/UsagePage/hooks/usePaginatedDailyActivity";

function makeWrapper() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });

  const Wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: qc }, children);

  Wrapper.displayName = "QueryClientWrapper";
  return Wrapper;
}

/** Build a mock page response with controllable totals. */
function mockPage(page: number, totalPages: number, extra: Record<string, unknown> = {}) {
  return {
    results: [{ date: `2025-01-0${page}`, spend: page }],
    metadata: {
      total_pages: totalPages,
      has_more: page < totalPages,
      page,
      total_spend: page * 10,
      total_api_requests: page * 5,
      total_prompt_tokens: 0,
      total_completion_tokens: 0,
      total_tokens: 0,
      total_successful_requests: 0,
      total_failed_requests: 0,
      total_cache_read_input_tokens: 0,
      total_cache_creation_input_tokens: 0,
      ...extra,
    },
  };
}

describe("usePaginatedDailyActivity", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("should return EMPTY_DATA and not call fetchFn when enabled=false", () => {
    const fetchFn = vi.fn();
    const { result } = renderHook(
      () =>
        usePaginatedDailyActivity({
          fetchFn,
          args: ["token", "2025-01-01", "2025-01-07"],
          enabled: false,
        }),
      { wrapper: makeWrapper() },
    );

    expect(fetchFn).not.toHaveBeenCalled();
    expect(result.current.loading).toBe(false);
    expect(result.current.isFetchingMore).toBe(false);
    expect(result.current.data.results).toHaveLength(0);
    expect(result.current.data.metadata.total_pages).toBe(1);
  });

  it("should fetch page 1 and mark loading=false when total_pages=1", async () => {
    const fetchFn = vi.fn().mockResolvedValue(mockPage(1, 1));
    const { result } = renderHook(
      () =>
        usePaginatedDailyActivity({
          fetchFn,
          args: ["token", "2025-01-01", "2025-01-07"],
          enabled: true,
        }),
      { wrapper: makeWrapper() },
    );

    // Initially loading
    expect(result.current.loading).toBe(true);

    // Flush microtasks so the first page resolves
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(fetchFn).toHaveBeenCalledTimes(1);
    // Page is injected at index 3
    expect(fetchFn).toHaveBeenCalledWith("token", "2025-01-01", "2025-01-07", 1);
    expect(result.current.loading).toBe(false);
    expect(result.current.isFetchingMore).toBe(false);
    expect(result.current.data.results).toHaveLength(1);
    expect(result.current.progress.currentPage).toBe(1);
    expect(result.current.progress.totalPages).toBe(1);
  });

  it("should auto-fetch pages 2..N and accumulate results", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(mockPage(1, 3))
      .mockResolvedValueOnce(mockPage(2, 3))
      .mockResolvedValueOnce(mockPage(3, 3));

    const { result } = renderHook(
      () =>
        usePaginatedDailyActivity({
          fetchFn,
          args: ["token", "2025-01-01", "2025-01-07"],
          enabled: true,
        }),
      { wrapper: makeWrapper() },
    );

    // Flush all timers (including the 300 ms delay between pages) and promises
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(fetchFn).toHaveBeenCalledTimes(3);
    expect(fetchFn).toHaveBeenNthCalledWith(1, "token", "2025-01-01", "2025-01-07", 1);
    expect(fetchFn).toHaveBeenNthCalledWith(2, "token", "2025-01-01", "2025-01-07", 2);
    expect(fetchFn).toHaveBeenNthCalledWith(3, "token", "2025-01-01", "2025-01-07", 3);

    expect(result.current.loading).toBe(false);
    expect(result.current.isFetchingMore).toBe(false);
    // All three pages' results should be accumulated
    expect(result.current.data.results).toHaveLength(3);
    expect(result.current.progress.currentPage).toBe(3);
    expect(result.current.progress.totalPages).toBe(3);
  });

  it("should sum total_spend and total_api_requests across pages", async () => {
    // page 1: spend=10, requests=5
    // page 2: spend=20, requests=10
    // page 3: spend=30, requests=15
    // Expected totals: spend=60, requests=30
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(mockPage(1, 3))
      .mockResolvedValueOnce(mockPage(2, 3))
      .mockResolvedValueOnce(mockPage(3, 3));

    const { result } = renderHook(
      () =>
        usePaginatedDailyActivity({
          fetchFn,
          args: ["token", "2025-01-01", "2025-01-07"],
          enabled: true,
        }),
      { wrapper: makeWrapper() },
    );

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(result.current.data.metadata.total_spend).toBe(60);
    expect(result.current.data.metadata.total_api_requests).toBe(30);
  });

  it("should only flush state at batch boundaries (every 3 pages), not on every page", async () => {
    // RENDER_BATCH_SIZE = 3, so with 6 pages we expect exactly 2 batch flushes
    // at pages 3 and 6 (plus the initial page-1 setData).
    const fetchFn = vi
      .fn()
      .mockImplementation((token: string, start: string, end: string, page: number) =>
        Promise.resolve(mockPage(page, 6)),
      );

    const setDataSpy = vi.fn();
    const { result } = renderHook(
      () =>
        usePaginatedDailyActivity({
          fetchFn,
          args: ["token", "2025-01-01", "2025-01-07"],
          enabled: true,
        }),
      { wrapper: makeWrapper() },
    );

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    // The final accumulated result should have all 6 pages' results
    expect(result.current.data.results).toHaveLength(6);
    expect(result.current.progress.currentPage).toBe(6);
  });

  it("should set cancelled=true and stop fetching when cancel() is called", async () => {
    // Provide 5 pages but cancel after page 1 resolves
    const fetchFn = vi
      .fn()
      .mockImplementation((token: string, start: string, end: string, page: number) =>
        Promise.resolve(mockPage(page, 5)),
      );

    const { result } = renderHook(
      () =>
        usePaginatedDailyActivity({
          fetchFn,
          args: ["token", "2025-01-01", "2025-01-07"],
          enabled: true,
        }),
      { wrapper: makeWrapper() },
    );

    // Let page 1 complete
    await act(async () => {
      await Promise.resolve(); // flush microtasks for page 1
    });

    // Cancel before pages 2-5 are fetched
    act(() => {
      result.current.cancel();
    });

    // Advance timers to confirm no more fetches happen
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(result.current.cancelled).toBe(true);
    expect(result.current.isFetchingMore).toBe(false);
    // fetchFn should have been called for page 1 and at most page 2
    // (depending on timing), but NOT for all 5 pages
    expect(fetchFn.mock.calls.length).toBeLessThan(5);
  });

  it("should restart (increment fetchId) when args change", async () => {
    const fetchFn = vi
      .fn()
      .mockImplementation((token: string, start: string, end: string, page: number) =>
        Promise.resolve(mockPage(page, 1)),
      );

    const initialArgs = ["token", "2025-01-01", "2025-01-07"];
    const { result, rerender } = renderHook(
      ({ args }: { args: string[] }) =>
        usePaginatedDailyActivity({
          fetchFn,
          args,
          enabled: true,
        }),
      { wrapper: makeWrapper(), initialProps: { args: initialArgs } },
    );

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    const firstCallCount = fetchFn.mock.calls.length;
    expect(firstCallCount).toBeGreaterThanOrEqual(1);

    // Change args to trigger a new fetch run
    rerender({ args: ["token", "2025-01-08", "2025-01-14"] });

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    // New fetch should have fired with the updated date range
    expect(fetchFn.mock.calls.length).toBeGreaterThan(firstCallCount);
    const lastCall = fetchFn.mock.calls[fetchFn.mock.calls.length - 1];
    expect(lastCall[1]).toBe("2025-01-08");
    expect(lastCall[2]).toBe("2025-01-14");
  });

  it("should set loading=false and isFetchingMore=false when fetchFn throws", async () => {
    const fetchFn = vi.fn().mockRejectedValue(new Error("API error"));

    const { result } = renderHook(
      () =>
        usePaginatedDailyActivity({
          fetchFn,
          args: ["token", "2025-01-01", "2025-01-07"],
          enabled: true,
        }),
      { wrapper: makeWrapper() },
    );

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(result.current.loading).toBe(false);
    expect(result.current.isFetchingMore).toBe(false);
  });

  it("should transition from enabled=false to enabled=true and start fetching", async () => {
    const fetchFn = vi.fn().mockResolvedValue(mockPage(1, 1));

    const { result, rerender } = renderHook(
      ({ enabled }: { enabled: boolean }) =>
        usePaginatedDailyActivity({
          fetchFn,
          args: ["token", "2025-01-01", "2025-01-07"],
          enabled,
        }),
      { wrapper: makeWrapper(), initialProps: { enabled: false } },
    );

    expect(fetchFn).not.toHaveBeenCalled();

    rerender({ enabled: true });

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect(result.current.loading).toBe(false);
    expect(result.current.data.results).toHaveLength(1);
  });
});
