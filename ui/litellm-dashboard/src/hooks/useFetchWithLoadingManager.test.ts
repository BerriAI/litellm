import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useFetchWithLoadingManager } from "./useFetchWithLoadingManager";

/**
 * Script scenario (generate_spend_data.sh): 1 call → wait 2s → 3 calls → wait 3s → 10 calls.
 * The loader reacts to Usage page fetches (poll), not chat completions.
 * This test simulates poll timing: requestFetch at t=0, t=2s, t=5s (matching script waves).
 */
describe("useFetchWithLoadingManager", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it("should match script timing: 1 call, 2s, 3 calls, 3s, 10 calls - loader on each fetch", async () => {
    const resolvers: Array<(v: { data: string }) => void> = [];
    const fetchFn = vi.fn().mockImplementation(
      () =>
        new Promise<{ data: string }>((r) => {
          resolvers.push(r);
        })
    );
    const { result } = renderHook(() =>
      useFetchWithLoadingManager(fetchFn, { debounceMs: 5000 })
    );

    // t=0: First requestFetch (like poll after initial load)
    act(() => {
      result.current.requestFetch();
    });
    expect(result.current.loading).toBe(true);
    expect(fetchFn).toHaveBeenCalledTimes(1);

    act(() => {
      resolvers[0]!({ data: "ok" });
    });
    await act(async () => {
      await vi.runAllTimersAsync();
    });
    expect(result.current.loading).toBe(false);

    // t=2s: Second requestFetch (within 5s - should debounce, loader shows immediately)
    await act(async () => {
      vi.advanceTimersByTimeAsync(2000);
    });
    act(() => {
      result.current.requestFetch();
    });
    expect(result.current.loading).toBe(true);
    expect(fetchFn).toHaveBeenCalledTimes(1);

    // Advance 3s (debounce delay) - debounced fetch runs, loader on
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(result.current.loading).toBe(true);
    expect(fetchFn).toHaveBeenCalledTimes(2);

    act(() => {
      resolvers[1]!({ data: "ok" });
    });
    await act(async () => {
      await vi.runAllTimersAsync();
    });
    expect(result.current.loading).toBe(false);

    // t=5s from last: Third requestFetch (5+ s since last - run immediately)
    await act(async () => {
      vi.advanceTimersByTimeAsync(5000);
    });
    act(() => {
      result.current.requestFetch();
    });
    expect(result.current.loading).toBe(true);
    expect(fetchFn).toHaveBeenCalledTimes(3);
  });

  it("should show loading on first request", async () => {
    const fetchFn = vi.fn().mockResolvedValue(undefined);
    const { result } = renderHook(() => useFetchWithLoadingManager(fetchFn));

    expect(result.current.loading).toBe(false);

    act(() => {
      result.current.requestFetch();
    });

    expect(result.current.loading).toBe(true);
    expect(fetchFn).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(result.current.loading).toBe(false);
  });

  it("should debounce requests within 5 seconds but show loader immediately", async () => {
    const fetchFn = vi.fn().mockResolvedValue(undefined);
    const { result } = renderHook(() => useFetchWithLoadingManager(fetchFn, { debounceMs: 5000 }));

    act(() => {
      result.current.requestFetch();
    });
    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect(result.current.loading).toBe(true);

    await act(async () => {
      await vi.runAllTimersAsync();
    });
    expect(result.current.loading).toBe(false);

    act(() => {
      result.current.requestFetch();
    });
    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect(result.current.loading).toBe(true);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("should debounce when requestFetch called while fetch is in progress", async () => {
    const resolvers: Array<(v: unknown) => void> = [];
    const fetchFn = vi.fn().mockImplementation(
      () =>
        new Promise((r) => {
          resolvers.push(r);
        })
    );
    const { result } = renderHook(() =>
      useFetchWithLoadingManager(fetchFn, { debounceMs: 5000 })
    );

    act(() => {
      result.current.requestFetch();
    });
    expect(result.current.loading).toBe(true);
    expect(fetchFn).toHaveBeenCalledTimes(1);

    // While first fetch is in progress, call requestFetch again - should debounce, loader stays on
    act(() => {
      result.current.requestFetch();
    });
    expect(result.current.loading).toBe(true);
    expect(fetchFn).toHaveBeenCalledTimes(1);

    act(() => {
      resolvers[0]!(undefined);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current.loading).toBe(true);

    // Debounced fetch should run 5s from when second requestFetch was called
    // lastFetchEnd is now set. timeSinceLastFetch at call time was ~0 (fetch was in progress).
    // So we scheduled for 5s. Advance 5s.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(fetchFn).toHaveBeenCalledTimes(2);
    expect(result.current.loading).toBe(true);

    act(() => {
      resolvers[1]!(undefined);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current.loading).toBe(false);
  });

  it("should never show loading when no fetch is in progress", async () => {
    const fetchFn = vi.fn().mockResolvedValue(undefined);
    const { result } = renderHook(() => useFetchWithLoadingManager(fetchFn));

    expect(result.current.loading).toBe(false);

    act(() => {
      result.current.requestFetch();
    });
    expect(result.current.loading).toBe(true);

    await act(async () => {
      await vi.runAllTimersAsync();
    });
    expect(result.current.loading).toBe(false);
  });
});
