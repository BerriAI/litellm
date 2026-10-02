import { QueryClient, QueryClientProvider, onlineManager, useQuery, skipToken } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { PropsWithChildren } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { isQueryPending } from "./queryReadiness";

const renderQuery = (enabled = true) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  const fetch = vi.fn().mockResolvedValue(["row"]);
  const wrapper = ({ children }: PropsWithChildren) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  const hook = renderHook(() => useQuery({ queryKey: ["readiness"], queryFn: enabled ? fetch : skipToken }), {
    wrapper,
  });
  return { ...hook, fetch, client };
};

afterEach(() => onlineManager.setOnline(true));

describe("query readiness", () => {
  it("waits for an offline first load and completes after reconnection", async () => {
    onlineManager.setOnline(false);
    const { result, fetch, unmount } = renderQuery();
    expect(isQueryPending(result.current)).toBe(true);
    expect(result.current.isPaused).toBe(true);
    expect(fetch).not.toHaveBeenCalled();

    act(() => onlineManager.setOnline(true));
    await waitFor(() => expect(result.current.data).toEqual(["row"]));
    expect(isQueryPending(result.current)).toBe(false);
    unmount();
  });

  it("does not wait for a disabled query while offline", () => {
    onlineManager.setOnline(false);
    const { result, fetch, unmount } = renderQuery(false);
    expect(isQueryPending(result.current)).toBe(false);
    expect(fetch).not.toHaveBeenCalled();
    unmount();
  });

  it("keeps cached rows ready while an offline refresh is paused", async () => {
    const { result, client, unmount } = renderQuery();
    await waitFor(() => expect(result.current.data).toEqual(["row"]));
    expect(result.current.isPaused).toBe(false);
    act(() => onlineManager.setOnline(false));
    act(() => {
      void client.invalidateQueries({ queryKey: ["readiness"] });
    });
    await waitFor(() => expect(result.current.isPaused).toBe(true));
    expect(isQueryPending(result.current)).toBe(false);
    expect(result.current.data).toEqual(["row"]);
    unmount();
  });
});
