import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  type DailyActivityAggregatedResponse,
  type DailyActivityRequest,
} from "@/components/UsagePage/dailyActivityApi";

vi.mock("@/components/networking", () => ({
  dailyActivityAggregatedCall: vi.fn(),
}));

import { dailyActivityAggregatedCall } from "@/components/networking";
import { useAggregatedDailyActivity } from "./dailyActivityQueries";

const mockAggregatedCall = vi.mocked(dailyActivityAggregatedCall);

const response = (spend: number): DailyActivityAggregatedResponse => ({
  results: [],
  metadata: { ...EMPTY_DAILY_ACTIVITY_METADATA, total_spend: spend },
});

const request = (entityId: string | null = null): DailyActivityRequest => ({
  accessToken: "token",
  startTime: new Date(2025, 0, 1),
  endTime: new Date(2025, 0, 31),
  entityIds: entityId ? [entityId] : null,
});

const createWrapper = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const Wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  return Wrapper;
};

describe("useAggregatedDailyActivity", () => {
  beforeEach(() => {
    mockAggregatedCall.mockReset();
  });
  it("fetches once per distinct request and refetches when the request changes", async () => {
    mockAggregatedCall.mockResolvedValue(response(12));
    const { result, rerender } = renderHook(
      ({ req }: { req: DailyActivityRequest | null }) => useAggregatedDailyActivity("user", req),
      { initialProps: { req: request() }, wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(mockAggregatedCall).toHaveBeenCalledTimes(1);
    expect(result.current.data?.metadata?.total_spend).toBe(12);

    rerender({ req: request("u1") });
    await waitFor(() => expect(mockAggregatedCall).toHaveBeenCalledTimes(2));
  });

  it("does not fetch for a null request and is not loading", () => {
    const { result } = renderHook(() => useAggregatedDailyActivity("user", null), { wrapper: createWrapper() });

    expect(mockAggregatedCall).not.toHaveBeenCalled();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.isError).toBe(false);
    expect(result.current.data).toBeUndefined();
  });

  it("exposes isError on rejection", async () => {
    mockAggregatedCall.mockRejectedValue(new Error("boom"));
    const { result } = renderHook(() => useAggregatedDailyActivity("user", request()), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.isLoading).toBe(false);
  });

  it("never shows a stale resolution from the previous request", async () => {
    let resolveFirst: ((value: DailyActivityAggregatedResponse) => void) | undefined;
    const first = new Promise<DailyActivityAggregatedResponse>((resolve) => {
      resolveFirst = resolve;
    });
    mockAggregatedCall.mockImplementationOnce(() => first).mockResolvedValueOnce(response(99));

    const { result, rerender } = renderHook(
      ({ entityId }: { entityId: string }) => useAggregatedDailyActivity("user", request(entityId)),
      { initialProps: { entityId: "a" }, wrapper: createWrapper() },
    );

    rerender({ entityId: "b" });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.data?.metadata?.total_spend).toBe(99);

    await act(async () => {
      resolveFirst?.(response(1));
    });
    expect(result.current.data?.metadata?.total_spend).toBe(99);
  });

  it("keeps the team excludeEntityIds default on the wire", async () => {
    mockAggregatedCall.mockResolvedValue(response(0));
    const { result } = renderHook(() => useAggregatedDailyActivity("team", request()), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(mockAggregatedCall).toHaveBeenCalledWith(
      "team",
      expect.objectContaining({ excludeEntityIds: ["litellm-dashboard"] }),
    );
  });
});
