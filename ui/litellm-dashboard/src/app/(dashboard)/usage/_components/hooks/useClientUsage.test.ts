import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { uiSpendLogsCall } from "@/components/networking";
import { UsageClient } from "./clientUsageData";
import { useClientUsage } from "./useClientUsage";

vi.mock("@/components/networking", () => ({ uiSpendLogsCall: vi.fn() }));
const options = {
  accessToken: "test-token",
  userId: "user-1",
  client: UsageClient.ClaudeDesktop,
  startTime: new Date("2026-10-22T07:00:00.500Z"),
  endTime: new Date("2026-10-23T06:59:59.999Z"),
};
const emptyPage = { data: [], total: 0, page: 1, total_pages: 0, total_is_capped: false };

describe("useClientUsage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should send a bounded authenticated user query with UTC bounds covering the selected interval", async () => {
    vi.mocked(uiSpendLogsCall).mockResolvedValue(emptyPage);
    const { result } = renderHook(() => useClientUsage(options));
    await waitFor(() => expect(result.current.loading).toBe(false));
    const expectedRequest = {
      accessToken: "test-token",
      start_date: "2026-10-22 07:00:00",
      end_date: "2026-10-23 07:00:00",
      page: 1,
      page_size: 1000,
      params: { user_id: "user-1", sort_by: "startTime", sort_order: "desc", group_by_session: false },
    };
    expect(uiSpendLogsCall).toHaveBeenCalledExactlyOnceWith(expectedRequest);
    expect(result.current.error).toBeNull();
  });

  it("should require a selected user and avoid querying the all-clients view", () => {
    const { result, rerender } = renderHook(
      ({ userId, client }: { userId: string | null; client: UsageClient | null }) =>
        useClientUsage({ ...options, userId, client }),
      { initialProps: { userId: null, client: UsageClient.ClaudeDesktop } },
    );
    expect(result.current.error).toBe("Select a user or team to view client usage");
    expect(result.current.loading).toBe(false);
    rerender({ userId: "user-1", client: null });
    expect(result.current.error).toBeNull();
    expect(uiSpendLogsCall).not.toHaveBeenCalled();
  });

  it("should expose incomplete results as an error without loading more pages", async () => {
    vi.mocked(uiSpendLogsCall).mockResolvedValue({ ...emptyPage, total: 1001, total_pages: 2 });
    const { result } = renderHook(() => useClientUsage(options));
    await waitFor(() => expect(result.current.error).toContain("Narrow the date range"));
    expect(result.current.data.results).toEqual([]);
    expect(uiSpendLogsCall).toHaveBeenCalledTimes(1);
  });

  it("should expose network failures", async () => {
    vi.mocked(uiSpendLogsCall).mockRejectedValue(new Error("No permission to view these logs"));
    const { result } = renderHook(() => useClientUsage(options));
    await waitFor(() => expect(result.current.error).toBe("No permission to view these logs"));
    expect(result.current.loading).toBe(false);
  });

  it("should ignore a stale response after the user changes", async () => {
    let resolveFirst: ((value: unknown) => void) | undefined;
    const first = new Promise<unknown>((resolve) => {
      resolveFirst = resolve;
    });
    vi.mocked(uiSpendLogsCall).mockReturnValueOnce(first).mockResolvedValueOnce(emptyPage);
    const { result, rerender } = renderHook(({ userId }) => useClientUsage({ ...options, userId }), {
      initialProps: { userId: "user-1" },
    });
    rerender({ userId: "user-2" });
    await waitFor(() => expect(result.current.loading).toBe(false));
    await act(async () => {
      resolveFirst?.({ ...emptyPage, total: 1001, total_pages: 2 });
    });
    expect(result.current.error).toBeNull();
    expect(result.current.data.metadata?.total_spend).toBe(0);
  });

  it("should clear loaded data immediately when client filtering is disabled", async () => {
    vi.mocked(uiSpendLogsCall).mockResolvedValue({ ...emptyPage, total: 1001, total_pages: 2 });
    const { result, rerender } = renderHook(
      ({ client }: { client: UsageClient | null }) => useClientUsage({ ...options, client }),
      { initialProps: { client: UsageClient.ClaudeDesktop } },
    );
    await waitFor(() => expect(result.current.error).toContain("Narrow the date range"));
    rerender({ client: null });
    expect(result.current.error).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.data.results).toEqual([]);
  });

  it("should retry a failed request with the same authenticated user and dates", async () => {
    vi.mocked(uiSpendLogsCall).mockRejectedValueOnce(new Error("Logs unavailable")).mockResolvedValueOnce(emptyPage);
    const { result } = renderHook(() => useClientUsage(options));
    await waitFor(() => expect(result.current.error).toBe("Logs unavailable"));

    act(() => result.current.reload());
    expect(result.current.error).toBeNull();
    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(uiSpendLogsCall).toHaveBeenCalledTimes(2);
    expect(vi.mocked(uiSpendLogsCall).mock.calls[1]).toEqual(vi.mocked(uiSpendLogsCall).mock.calls[0]);
  });

  it("should hide previous person totals until a reload completes", async () => {
    const row = {
      request_id: "request-1",
      user: options.userId,
      startTime: "2026-10-22T12:00:00Z",
      spend: 3,
      total_tokens: 100,
      prompt_tokens: 80,
      completion_tokens: 20,
      request_tags: [],
      metadata: { user_agent: "claude-desktop/1.0" },
    };
    const page = { ...emptyPage, data: [row], total: 1, total_pages: 1 };
    vi.mocked(uiSpendLogsCall)
      .mockResolvedValueOnce(page)
      .mockReturnValueOnce(new Promise(() => {}));
    const { result } = renderHook(() => useClientUsage(options));
    await waitFor(() => expect(result.current.data.metadata?.total_spend).toBe(3));

    act(() => result.current.reload());
    expect(result.current.data.results).toEqual([]);
    expect(result.current.loading).toBe(true);
  });
  it("should query team totals and narrow to a member with both filters", async () => {
    vi.mocked(uiSpendLogsCall).mockResolvedValue(emptyPage);
    const initialProps: { userId: string | null } = { userId: null };
    const { result, rerender } = renderHook(
      ({ userId }: { userId: string | null }) => useClientUsage({ ...options, userId, teamId: "engineering" }),
      { initialProps },
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    const teamParams = {
      team_id: "engineering",
      sort_by: "startTime",
      sort_order: "desc",
      group_by_session: false,
    };
    expect(vi.mocked(uiSpendLogsCall).mock.calls[0][0].params).toEqual(teamParams);
    rerender({ userId: "user-1" });
    await waitFor(() => expect(result.current.loading).toBe(false));
    const memberParams = { ...teamParams, user_id: "user-1" };
    expect(vi.mocked(uiSpendLogsCall).mock.calls[1][0].params).toEqual(memberParams);
  });

  it("should ignore an old response when switching teams", async () => {
    let resolveFirst: ((response: unknown) => void) | undefined;
    const first = new Promise<unknown>((resolve) => {
      resolveFirst = resolve;
    });
    vi.mocked(uiSpendLogsCall).mockReturnValueOnce(first).mockResolvedValueOnce(emptyPage);
    const { result, rerender } = renderHook(({ teamId }) => useClientUsage({ ...options, teamId, userId: null }), {
      initialProps: { teamId: "engineering" },
    });
    rerender({ teamId: "design" });
    await waitFor(() => expect(result.current.loading).toBe(false));
    await act(async () => resolveFirst?.({ ...emptyPage, total: 1001, total_pages: 2 }));
    expect(result.current.error).toBeNull();
    expect(result.current.data.metadata?.total_spend).toBe(0);
  });
});
