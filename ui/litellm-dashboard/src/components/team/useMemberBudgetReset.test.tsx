import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useMemberBudgetReset, type MemberBudgetResetGateway } from "./useMemberBudgetReset";

const pending = {
  teamId: "team-1",
  updateData: { team_id: "team-1", team_member_budget: 100 },
  memberCount: 3,
  newBudget: 100,
};

describe("useMemberBudgetReset", () => {
  it("blocks repeated saves until the whole transaction finishes", async () => {
    const transaction = Promise.withResolvers<{ member_budgets_updated: number }>();
    const gateway: MemberBudgetResetGateway = {
      saveTeam: vi.fn(() => transaction.promise),
      refreshTeamData: vi.fn().mockResolvedValue(undefined),
    };
    const { result } = renderHook(() => useMemberBudgetReset(gateway));
    act(() => result.current.prompt(pending));
    act(() => {
      void result.current.save("both");
      void result.current.save("both");
    });
    expect(gateway.saveTeam).toHaveBeenCalledTimes(1);
    expect(result.current.state.phase).toBe("saving");
    expect(gateway.refreshTeamData).not.toHaveBeenCalled();
    await act(async () => transaction.resolve({ member_budgets_updated: 2 }));
    expect(result.current.state.phase).toBe("idle");
    expect(gateway.refreshTeamData).toHaveBeenCalledTimes(1);
  });

  it("ignores completion of an old team's save after a new prompt opens", async () => {
    const transaction = Promise.withResolvers<{ member_budgets_updated: number }>();
    const gateway: MemberBudgetResetGateway = {
      saveTeam: vi.fn(() => transaction.promise),
      refreshTeamData: vi.fn().mockResolvedValue(undefined),
    };
    const { result } = renderHook(() => useMemberBudgetReset(gateway));
    act(() => result.current.prompt(pending));
    act(() => {
      void result.current.save("raise");
    });
    act(() => result.current.dismiss());
    const next = { ...pending, teamId: "team-2", updateData: { ...pending.updateData, team_id: "team-2" } };
    act(() => result.current.prompt(next));
    await act(async () => transaction.resolve({ member_budgets_updated: 2 }));
    expect(result.current.state).toEqual({ phase: "prompting", pending: next });
    expect(gateway.refreshTeamData).not.toHaveBeenCalled();
  });
});
