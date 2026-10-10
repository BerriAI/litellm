import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createElement, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { teamInfoCall } from "@/components/networking";
import type { TeamMemberBudgetSource } from "@/components/shared/InheritedBudgetHint";
import { teamMemberBudgetGate } from "@/components/shared/InheritedBudgetHint";
import { teamMemberBudgetQueryKey, useTeamMemberBudgets } from "./useTeamMemberBudgets";

const authMocks = vi.hoisted(() => ({
  useAuthorized: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: authMocks.useAuthorized,
}));

vi.mock("@/components/networking", async (importOriginal) => {
  const networking = await importOriginal<typeof import("@/components/networking")>();
  return { ...networking, teamInfoCall: vi.fn() };
});

describe("useTeamMemberBudgets", () => {
  const teamId = "team-1";
  const teamIds = [teamId];
  const teamInfo: TeamMemberBudgetSource = {
    team_info: { team_id: teamId },
  };
  beforeEach(() => {
    vi.clearAllMocks();
    authMocks.useAuthorized.mockReturnValue({ accessToken: "access-token", userId: "viewer-one" });
    vi.mocked(teamInfoCall).mockResolvedValue(teamInfo as never);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  const createQueryWrapper = () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });
    const wrapper = ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client: queryClient }, children);
    return { queryClient, wrapper };
  };

  it("keeps separate cached results for different viewers", async () => {
    const { queryClient, wrapper } = createQueryWrapper();
    const viewerOneKey = [...teamMemberBudgetQueryKey(teamId), "viewer-one"];
    const viewerTwoKey = [...teamMemberBudgetQueryKey(teamId), "viewer-two"];

    const firstViewer = renderHook(() => useTeamMemberBudgets(teamIds), { wrapper });
    await waitFor(() => expect(teamInfoCall).toHaveBeenCalledTimes(1));
    firstViewer.unmount();

    authMocks.useAuthorized.mockReturnValue({ accessToken: "access-token", userId: "viewer-two" });
    renderHook(() => useTeamMemberBudgets(teamIds), { wrapper });
    await waitFor(() => expect(teamInfoCall).toHaveBeenCalledTimes(2));

    const cachedKeys = queryClient
      .getQueryCache()
      .findAll({ queryKey: teamMemberBudgetQueryKey(teamId) })
      .map((query) => query.queryKey);
    expect(cachedKeys).toHaveLength(2);
    expect(cachedKeys).toContainEqual(viewerOneKey);
    expect(cachedKeys).toContainEqual(viewerTwoKey);
    expect(teamInfoCall).toHaveBeenNthCalledWith(1, "access-token", teamId, { keyLimit: 1 });
    expect(teamInfoCall).toHaveBeenNthCalledWith(2, "access-token", teamId, { keyLimit: 1 });
  });

  it("refetches a viewer-scoped entry when invalidated by the team prefix", async () => {
    const { queryClient, wrapper } = createQueryWrapper();
    renderHook(() => useTeamMemberBudgets(teamIds), { wrapper });
    await waitFor(() => expect(teamInfoCall).toHaveBeenCalledTimes(1));

    await act(async () => {
      await queryClient.invalidateQueries({ queryKey: teamMemberBudgetQueryKey(teamId) });
    });

    expect(teamInfoCall).toHaveBeenCalledTimes(2);
  });

  it("drops an expired member budget increase without refetching", async () => {
    vi.useFakeTimers({ now: new Date("2026-06-20T12:00:00Z") });
    const expiry = new Date(Date.now() + 60_000).toISOString();
    vi.mocked(teamInfoCall).mockResolvedValue({
      team_info: { team_id: teamId },
      team_memberships: [
        {
          user_id: "viewer-one",
          litellm_budget_table: {
            max_budget: 50,
            temp_budget_increase: 25,
            temp_budget_expiry: expiry,
          },
        },
      ],
    } as never);

    const { wrapper } = createQueryWrapper();
    const { result } = renderHook(
      () => teamMemberBudgetGate(useTeamMemberBudgets(teamIds)[teamId], "viewer-one")?.maxBudget,
      { wrapper },
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current).toBe(75);

    act(() => {
      vi.advanceTimersByTime(61_000);
    });

    expect(result.current).toBe(50);
    expect(teamInfoCall).toHaveBeenCalledTimes(1);
  });

  it("re-arms the timer for the next expiry after one fires", async () => {
    vi.useFakeTimers({ now: new Date("2026-06-20T12:00:00Z") });
    const teamTwoId = "team-2";
    const membershipFor = (increase: number, expiryOffsetMs: number) => ({
      team_info: { team_id: "ignored" },
      team_memberships: [
        {
          user_id: "viewer-one",
          litellm_budget_table: {
            max_budget: 50,
            temp_budget_increase: increase,
            temp_budget_expiry: new Date(Date.now() + expiryOffsetMs).toISOString(),
          },
        },
      ],
    });
    vi.mocked(teamInfoCall).mockImplementation(async (_token, id) =>
      id === teamId ? (membershipFor(25, 60_000) as never) : (membershipFor(10, 120_000) as never),
    );

    const { wrapper } = createQueryWrapper();
    const { result } = renderHook(
      () => {
        const budgets = useTeamMemberBudgets([teamId, teamTwoId]);
        return {
          teamOne: teamMemberBudgetGate(budgets[teamId], "viewer-one")?.maxBudget,
          teamTwo: teamMemberBudgetGate(budgets[teamTwoId], "viewer-one")?.maxBudget,
        };
      },
      { wrapper },
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current).toEqual({ teamOne: 75, teamTwo: 60 });

    act(() => {
      vi.advanceTimersByTime(61_000);
    });
    expect(result.current).toEqual({ teamOne: 50, teamTwo: 60 });

    act(() => {
      vi.advanceTimersByTime(60_000);
    });
    expect(result.current).toEqual({ teamOne: 50, teamTwo: 50 });
  });

  it("returns the same object reference when no increase is active", async () => {
    vi.useFakeTimers({ now: new Date("2026-06-20T12:00:00Z") });
    vi.mocked(teamInfoCall).mockResolvedValue({
      team_info: { team_id: teamId },
      team_memberships: [{ user_id: "viewer-one", litellm_budget_table: { max_budget: 50 } }],
    } as never);

    const { wrapper } = createQueryWrapper();
    const { result } = renderHook(() => useTeamMemberBudgets(teamIds), { wrapper });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    const before = result.current;

    act(() => {
      vi.advanceTimersByTime(86_400_000);
    });

    expect(result.current).toBe(before);
  });

  it("does not fire early when the next expiry is far in the future", async () => {
    vi.useFakeTimers({ now: new Date("2026-06-20T12:00:00Z") });
    const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout");
    const teamTwoId = "team-2";
    const membershipFor = (increase: number, expiryOffsetMs: number) => ({
      team_info: { team_id: "ignored" },
      team_memberships: [
        {
          user_id: "viewer-one",
          litellm_budget_table: {
            max_budget: 50,
            temp_budget_increase: increase,
            temp_budget_expiry: new Date(Date.now() + expiryOffsetMs).toISOString(),
          },
        },
      ],
    });
    vi.mocked(teamInfoCall).mockImplementation(async (_token, id) =>
      id === teamId ? (membershipFor(25, 40 * 86_400_000) as never) : (membershipFor(10, 41 * 86_400_000) as never),
    );

    const { wrapper } = createQueryWrapper();
    const { result } = renderHook(
      () => {
        const budgets = useTeamMemberBudgets([teamId, teamTwoId]);
        return {
          budgets,
          teamOne: teamMemberBudgetGate(budgets[teamId], "viewer-one")?.maxBudget,
          teamTwo: teamMemberBudgetGate(budgets[teamTwoId], "viewer-one")?.maxBudget,
        };
      },
      { wrapper },
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current).toMatchObject({ teamOne: 75, teamTwo: 60 });
    const before = result.current;

    act(() => {
      vi.advanceTimersByTime(1_000);
    });

    expect(result.current).toMatchObject({ teamOne: 75, teamTwo: 60 });
    expect(result.current).toBe(before);
    const scheduledDelays = setTimeoutSpy.mock.calls.map(([, delay]) => delay ?? 0);
    expect(Math.max(...scheduledDelays)).toBeLessThanOrEqual(2_147_483_647);

    act(() => {
      vi.advanceTimersByTime(25 * 86_400_000);
    });
    act(() => {
      vi.advanceTimersByTime(15.2 * 86_400_000);
    });
    expect(result.current).toMatchObject({ teamOne: 50, teamTwo: 60 });
    act(() => {
      vi.advanceTimersByTime(2 * 86_400_000);
    });
    expect(result.current).toMatchObject({ teamOne: 50, teamTwo: 50 });
  });
});
