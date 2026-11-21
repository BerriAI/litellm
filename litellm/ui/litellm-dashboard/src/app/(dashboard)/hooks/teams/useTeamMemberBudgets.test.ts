import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createElement, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { teamInfoCall } from "@/components/networking";
import type { TeamMemberBudgetSource } from "@/components/shared/InheritedBudgetHint";
import { teamMemberBudgetGate } from "@/components/shared/InheritedBudgetHint";
import { teamMemberBudgetQueryKey, useTeamMemberBudgets } from "./useTeamMemberBudgets";
import { fetchTeamMembersByUserId } from "./useTeams";

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

vi.mock("./useTeams", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./useTeams")>()),
  fetchTeamMembersByUserId: vi.fn(),
}));

describe("useTeamMemberBudgets", () => {
  const teamId = "team-1";
  const teamTwoId = "team-2";
  const userIdsByTeam = { [teamId]: ["viewer-one"] };
  const userIdsByBothTeams = { [teamId]: ["viewer-one"], [teamTwoId]: ["viewer-one"] };
  const teamInfo: TeamMemberBudgetSource = {
    team_info: { team_id: teamId },
  };
  const serve = (sourceFor: (teamId: string) => TeamMemberBudgetSource) => {
    vi.mocked(teamInfoCall).mockImplementation(
      async (_token, id) => ({ team_info: sourceFor(id as string).team_info, team_memberships: [] }) as never,
    );
    vi.mocked(fetchTeamMembersByUserId).mockImplementation(
      async (_token, id) => (sourceFor(id).team_memberships ?? []) as never,
    );
  };
  beforeEach(() => {
    vi.clearAllMocks();
    authMocks.useAuthorized.mockReturnValue({ accessToken: "access-token", userId: "viewer-one" });
    serve(() => teamInfo);
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
    const viewerOneKey = [...teamMemberBudgetQueryKey(teamId), "viewer-one", ["viewer-one"]];
    const viewerTwoKey = [...teamMemberBudgetQueryKey(teamId), "viewer-two", ["viewer-one"]];

    const firstViewer = renderHook(() => useTeamMemberBudgets(userIdsByTeam), { wrapper });
    await waitFor(() => expect(teamInfoCall).toHaveBeenCalledTimes(1));
    firstViewer.unmount();

    authMocks.useAuthorized.mockReturnValue({ accessToken: "access-token", userId: "viewer-two" });
    renderHook(() => useTeamMemberBudgets(userIdsByTeam), { wrapper });
    await waitFor(() => expect(teamInfoCall).toHaveBeenCalledTimes(2));

    const cachedKeys = queryClient
      .getQueryCache()
      .findAll({ queryKey: teamMemberBudgetQueryKey(teamId) })
      .map((query) => query.queryKey);
    expect(cachedKeys).toHaveLength(2);
    expect(cachedKeys).toContainEqual(viewerOneKey);
    expect(cachedKeys).toContainEqual(viewerTwoKey);
    expect(teamInfoCall).toHaveBeenNthCalledWith(1, "access-token", teamId, { includeMembers: false, keyLimit: 1 });
    expect(teamInfoCall).toHaveBeenNthCalledWith(2, "access-token", teamId, { includeMembers: false, keyLimit: 1 });
  });

  it("reads only the key owners' memberships and gates them like the whole roster did", async () => {
    const wholeRoster: TeamMemberBudgetSource = {
      team_info: { team_id: teamId, team_alias: "Platform", team_member_budget_table: { max_budget: 20 } },
      team_memberships: [
        { user_id: "owner-a", spend: 7, litellm_budget_table: { max_budget: 50, budget_duration: "30d" } },
        { user_id: "owner-b", spend: 3, litellm_budget_table: null },
        { user_id: "bystander", spend: 99, litellm_budget_table: { max_budget: 1 } },
      ],
    };
    vi.mocked(teamInfoCall).mockResolvedValue({ team_info: wholeRoster.team_info, team_memberships: [] } as never);
    vi.mocked(fetchTeamMembersByUserId).mockImplementation(
      async (_token, _teamId, userIds) =>
        [
          ...(wholeRoster.team_memberships ?? []).filter((row) => userIds.includes(row.user_id)),
          { user_id: null, spend: 0 },
        ] as never,
    );
    const owners = { [teamId]: ["owner-b", "owner-a", "owner-b", "owner-without-membership"] };

    const { wrapper } = createQueryWrapper();
    const { result } = renderHook(() => useTeamMemberBudgets(owners)[teamId], { wrapper });
    await waitFor(() => expect(result.current).toBeDefined());

    expect(fetchTeamMembersByUserId).toHaveBeenCalledExactlyOnceWith("access-token", teamId, [
      "owner-a",
      "owner-b",
      "owner-without-membership",
    ]);
    expect(result.current.team_memberships?.map((row) => row.user_id)).toEqual(["owner-a", "owner-b"]);
    const now = new Date("2026-06-20T12:00:00Z");
    for (const owner of ["owner-a", "owner-b", "owner-without-membership"]) {
      expect(teamMemberBudgetGate(result.current, owner, null, now)).toEqual(
        teamMemberBudgetGate(wholeRoster, owner, null, now),
      );
    }
    expect(teamMemberBudgetGate(result.current, "owner-a", null, now)?.maxBudget).toBe(50);
    expect(teamMemberBudgetGate(result.current, "owner-b", null, now)).toMatchObject({ maxBudget: 20, spend: 3 });
  });

  it("keeps the hint of an owner already on screen while another page of owners loads", async () => {
    const team = { team_id: teamId, team_member_budget_table: { max_budget: 20 } };
    const row = (userId: string) => ({ user_id: userId, spend: 7, litellm_budget_table: { max_budget: 50 } });
    vi.mocked(teamInfoCall).mockResolvedValue({ team_info: team, team_memberships: [] } as never);
    vi.mocked(fetchTeamMembersByUserId).mockResolvedValueOnce([row("owner-a")] as never);
    const { wrapper } = createQueryWrapper();
    const { result, rerender } = renderHook((owners: readonly string[]) => useTeamMemberBudgets({ [teamId]: owners }), {
      wrapper,
      initialProps: ["owner-a"],
    });
    await waitFor(() => expect(teamMemberBudgetGate(result.current[teamId], "owner-a")?.maxBudget).toBe(50));

    const nextPage = Promise.withResolvers<never>();
    vi.mocked(fetchTeamMembersByUserId).mockReturnValueOnce(nextPage.promise);
    rerender(["owner-a", "owner-b"]);

    expect(teamMemberBudgetGate(result.current[teamId], "owner-a")).toMatchObject({ maxBudget: 50, spend: 7 });
    expect(teamMemberBudgetGate(result.current[teamId], "owner-b")).toBeNull();

    nextPage.resolve([row("owner-a"), { user_id: "owner-b", spend: 3 }] as never);
    await waitFor(() =>
      expect(teamMemberBudgetGate(result.current[teamId], "owner-b")).toMatchObject({ maxBudget: 20, spend: 3 }),
    );
  });

  it("refetches a viewer-scoped entry when invalidated by the team prefix", async () => {
    const { queryClient, wrapper } = createQueryWrapper();
    renderHook(() => useTeamMemberBudgets(userIdsByTeam), { wrapper });
    await waitFor(() => expect(teamInfoCall).toHaveBeenCalledTimes(1));

    await act(async () => {
      await queryClient.invalidateQueries({ queryKey: teamMemberBudgetQueryKey(teamId) });
    });

    expect(teamInfoCall).toHaveBeenCalledTimes(2);
  });

  it("drops an expired member budget increase without refetching", async () => {
    vi.useFakeTimers({ now: new Date("2026-06-20T12:00:00Z") });
    const expiry = new Date(Date.now() + 60_000).toISOString();
    serve(() => ({
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
    }));

    const { wrapper } = createQueryWrapper();
    const { result } = renderHook(
      () => teamMemberBudgetGate(useTeamMemberBudgets(userIdsByTeam)[teamId], "viewer-one")?.maxBudget,
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
    serve((id) => (id === teamId ? membershipFor(25, 60_000) : membershipFor(10, 120_000)));

    const { wrapper } = createQueryWrapper();
    const { result } = renderHook(
      () => {
        const budgets = useTeamMemberBudgets(userIdsByBothTeams);
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
    serve(() => ({
      team_info: { team_id: teamId },
      team_memberships: [{ user_id: "viewer-one", litellm_budget_table: { max_budget: 50 } }],
    }));

    const { wrapper } = createQueryWrapper();
    const { result } = renderHook(() => useTeamMemberBudgets(userIdsByTeam), { wrapper });
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
    serve((id) => (id === teamId ? membershipFor(25, 40 * 86_400_000) : membershipFor(10, 41 * 86_400_000)));

    const { wrapper } = createQueryWrapper();
    const { result } = renderHook(
      () => {
        const budgets = useTeamMemberBudgets(userIdsByBothTeams);
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
