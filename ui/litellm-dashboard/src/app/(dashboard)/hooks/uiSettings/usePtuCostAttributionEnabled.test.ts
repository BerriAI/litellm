import { getUiSettings } from "@/components/networking";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PTU_FLAG_REFRESH_MS, usePtuCostAttributionEnabled } from "./usePtuCostAttributionEnabled";
import { useUISettings } from "./useUISettings";

vi.mock("@/components/networking", () => ({
  getUiSettings: vi.fn(),
}));

describe("usePtuCostAttributionEnabled", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    vi.clearAllMocks();
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  /** Read the flag alongside the query it derives from, so assertions wait for a settled fetch. */
  const renderSettledFlag = async (settings: unknown) => {
    (getUiSettings as any).mockResolvedValue(settings);
    const { result } = renderHook(() => ({ enabled: usePtuCostAttributionEnabled(), query: useUISettings() }), {
      wrapper,
    });
    await waitFor(() => {
      expect(result.current.query.isSuccess).toBe(true);
    });
    return result;
  };

  it("is true only when the proxy reports the flag as enabled", async () => {
    const result = await renderSettledFlag({ values: { enable_ptu_cost_attribution: true } });
    expect(result.current.enabled).toBe(true);
  });

  it("is false when the proxy reports the flag as disabled", async () => {
    const result = await renderSettledFlag({ values: { enable_ptu_cost_attribution: false } });
    expect(result.current.enabled).toBe(false);
  });

  it("is false when the proxy omits the flag entirely", async () => {
    const result = await renderSettledFlag({ values: { enable_chat_ui: true } });
    expect(result.current.enabled).toBe(false);
  });

  it("is false when the proxy returns no values at all", async () => {
    const result = await renderSettledFlag({});
    expect(result.current.enabled).toBe(false);
  });

  it("does not treat a truthy non-boolean as enabled", async () => {
    const result = await renderSettledFlag({ values: { enable_ptu_cost_attribution: "false" } });
    expect(result.current.enabled).toBe(false);
  });

  it("does not treat the string 'true' as enabled, since the proxy sends a real boolean", async () => {
    const result = await renderSettledFlag({ values: { enable_ptu_cost_attribution: "true" } });
    expect(result.current.enabled).toBe(false);
  });

  it("is false before the settings request resolves", () => {
    (getUiSettings as any).mockReturnValue(new Promise(() => {}));
    const { result } = renderHook(() => usePtuCostAttributionEnabled(), { wrapper });
    expect(result.current).toBe(false);
  });

  it("is false when the settings request fails", async () => {
    (getUiSettings as any).mockRejectedValue(new Error("boom"));
    const { result } = renderHook(() => ({ enabled: usePtuCostAttributionEnabled(), query: useUISettings() }), {
      wrapper,
    });
    await waitFor(() => {
      expect(result.current.query.isError).toBe(true);
    });
    expect(result.current.enabled).toBe(false);
  });
});

describe("staleness", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    vi.clearAllMocks();
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("re-reads the flag on its own cadence, not the shared one-hour default", async () => {
    // The other UI settings are persisted records. This one tracks the proxy process, so a
    // restart that flips the env var must not leave the model form offering PTU inputs the
    // backend now rejects for the rest of the hour.
    (getUiSettings as any).mockResolvedValue({ values: { enable_ptu_cost_attribution: true } });
    const { result } = renderHook(() => usePtuCostAttributionEnabled(), { wrapper });
    await waitFor(() => {
      expect(result.current).toBe(true);
    });

    const observers = queryClient.getQueryCache().getAll()[0].observers;
    expect(observers).toHaveLength(1);
    expect(observers[0].options.staleTime).toBe(PTU_FLAG_REFRESH_MS);
    expect(PTU_FLAG_REFRESH_MS).toBeLessThan(60 * 60 * 1000);
    // staleTime only marks the copy stale. A form that stays mounted and focused never
    // refetches on its own, so the flag has to be polled or it never notices a restart.
    expect(observers[0].options.refetchInterval).toBe(PTU_FLAG_REFRESH_MS);
  });

  it("leaves the default alone for every other settings consumer", async () => {
    (getUiSettings as any).mockResolvedValue({ values: {} });
    const { result } = renderHook(() => useUISettings(), { wrapper });
    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const observers = queryClient.getQueryCache().getAll()[0].observers;
    expect(observers[0].options.staleTime).toBe(60 * 60 * 1000);
    expect(observers[0].options.refetchInterval).toBeUndefined();
  });
});
