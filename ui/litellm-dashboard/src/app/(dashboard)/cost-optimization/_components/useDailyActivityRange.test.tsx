import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/networking", () => ({
  dailyActivityAggregatedCall: vi.fn().mockResolvedValue({ results: [], metadata: {} }),
}));

import { dailyActivityAggregatedCall } from "@/components/networking";
import { useActivityDateRange, useDailyActivityRange } from "./useDailyActivityRange";

const mockAggregatedCall = vi.mocked(dailyActivityAggregatedCall);

const createWrapper = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const Wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  return Wrapper;
};

describe("useDailyActivityRange", () => {
  beforeEach(() => {
    mockAggregatedCall.mockClear();
  });
  it("offers date-range state without starting a daily-activity query", () => {
    const { result } = renderHook(() => useActivityDateRange());

    expect(result.current.dateValue.from).toBeInstanceOf(Date);
    expect(result.current.dateValue.to).toBeInstanceOf(Date);
    expect(mockAggregatedCall).not.toHaveBeenCalled();
  });

  it("fetches every user's activity for an admin through the aggregated endpoint", async () => {
    const { result } = renderHook(() => useDailyActivityRange("test-token", "u1", "proxy_admin"), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(mockAggregatedCall).toHaveBeenCalledWith(
      "user",
      expect.objectContaining({
        accessToken: "test-token",
        entityIds: null,
        includeCurrentUtcDay: true,
      }),
    );
  });

  it("scopes the query to the caller for a non-admin", async () => {
    renderHook(() => useDailyActivityRange("test-token", "u1", "internal_user"), { wrapper: createWrapper() });

    await waitFor(() => expect(mockAggregatedCall).toHaveBeenCalled());
    expect(mockAggregatedCall).toHaveBeenCalledWith("user", expect.objectContaining({ entityIds: ["u1"] }));
  });

  it.each(["org_admin", "Org Admin"])(
    "scopes the query to the caller for %s, who has no admin view on this endpoint",
    async (role) => {
      renderHook(() => useDailyActivityRange("test-token", "u1", role), { wrapper: createWrapper() });

      await waitFor(() => expect(mockAggregatedCall).toHaveBeenCalled());
      expect(mockAggregatedCall).toHaveBeenCalledWith("user", expect.objectContaining({ entityIds: ["u1"] }));
    },
  );

  it("stays disabled until an access token is available", () => {
    const { result } = renderHook(() => useDailyActivityRange(null, "u1", "proxy_admin"), {
      wrapper: createWrapper(),
    });

    expect(result.current.loading).toBe(false);
    expect(mockAggregatedCall).not.toHaveBeenCalled();
  });

  it("exposes the request scope so sibling hooks fetch under the same filters", () => {
    const { result } = renderHook(() => useDailyActivityRange("test-token", "u1", "internal_user"), {
      wrapper: createWrapper(),
    });

    expect(result.current.scope).toMatchObject({
      accessToken: "test-token",
      userId: "u1",
      apiKey: null,
    });
  });
});
