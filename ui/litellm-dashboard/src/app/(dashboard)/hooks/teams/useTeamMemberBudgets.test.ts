import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createElement, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { teamInfoCall } from "@/components/networking";
import type { TeamMemberBudgetSource } from "@/components/shared/InheritedBudgetHint";
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
  let queryClient: QueryClient;

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });
    authMocks.useAuthorized.mockReturnValue({ accessToken: "access-token", userId: "viewer-one" });
    vi.mocked(teamInfoCall).mockResolvedValue(teamInfo as never);
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    createElement(QueryClientProvider, { client: queryClient }, children);

  it("keeps separate cached results for different viewers", async () => {
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
    renderHook(() => useTeamMemberBudgets(teamIds), { wrapper });
    await waitFor(() => expect(teamInfoCall).toHaveBeenCalledTimes(1));

    await act(async () => {
      await queryClient.invalidateQueries({ queryKey: teamMemberBudgetQueryKey(teamId) });
    });

    expect(teamInfoCall).toHaveBeenCalledTimes(2);
  });
});
