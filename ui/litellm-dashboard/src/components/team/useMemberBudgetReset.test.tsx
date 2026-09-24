import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { toast } from "@/lib/toast";
import { MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES } from "./memberBudgetReset";
import { useMemberBudgetReset, type MemberBudgetResetGateway } from "./useMemberBudgetReset";

const buildGateway = () => ({
  saveTeam: vi.fn(async (_updateData: Record<string, unknown>) => {}),
  resetMemberBudgets: vi.fn(async (_teamId: string, userIds: readonly string[]) =>
    userIds.map((user_id) => ({ success: true, user_id })),
  ),
  refreshTeamData: vi.fn(async () => {}),
});

const pendingFor = (userIds: string[]) => ({
  teamId: "team-123",
  updateData: { team_id: "team-123", team_member_budget: 20 },
  userIds,
  newBudget: 20,
});

const renderReset = (gateway: MemberBudgetResetGateway) => renderHook(() => useMemberBudgetReset(gateway));

describe("useMemberBudgetReset", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("starts idle and prompts with the pending update", async () => {
    const gateway = buildGateway();
    const { result } = renderReset(gateway);

    expect(result.current.state.phase).toBe("idle");

    await act(async () => {
      result.current.prompt(pendingFor(["u-1"]));
    });

    expect(result.current.state).toEqual({
      phase: "prompting",
      pending: pendingFor(["u-1"]),
    });
    expect(gateway.saveTeam).not.toHaveBeenCalled();
  });

  it("saves the team once, resets every member, and refreshes on success", async () => {
    const gateway = buildGateway();
    const { result } = renderReset(gateway);

    await act(async () => {
      result.current.prompt(pendingFor(["u-1", "u-2"]));
    });
    await act(async () => {
      await result.current.reset();
    });

    expect(gateway.saveTeam).toHaveBeenCalledTimes(1);
    expect(gateway.saveTeam).toHaveBeenCalledWith({ team_id: "team-123", team_member_budget: 20 });
    expect(gateway.resetMemberBudgets).toHaveBeenCalledWith("team-123", ["u-1", "u-2"]);
    expect(toast.success).toHaveBeenCalledWith("Reset 2 member budgets to the team default");
    expect(gateway.refreshTeamData).toHaveBeenCalledTimes(1);
    expect(result.current.state.phase).toBe("idle");
  });

  it("resets members in batches no larger than the bulk endpoint limit", async () => {
    const userIds = Array.from({ length: MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES + 1 }, (_, i) => `u-${i}`);
    const gateway = buildGateway();
    const { result } = renderReset(gateway);

    await act(async () => {
      result.current.prompt(pendingFor(userIds));
    });
    await act(async () => {
      await result.current.reset();
    });

    expect(gateway.resetMemberBudgets).toHaveBeenCalledTimes(2);
    expect(gateway.resetMemberBudgets.mock.calls[0][0]).toBe("team-123");
    expect(gateway.resetMemberBudgets.mock.calls[0][1]).toHaveLength(MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES);
    expect(gateway.resetMemberBudgets.mock.calls[1][1]).toEqual([`u-${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES}`]);
    expect(result.current.state.phase).toBe("idle");
  });

  it("resets against the team carried by the pending update", async () => {
    const gateway = buildGateway();
    const { result } = renderReset(gateway);

    const pending = {
      teamId: "team-999",
      updateData: { team_id: "team-999", team_member_budget: 20 },
      userIds: ["u-1"],
      newBudget: 20,
    };
    await act(async () => {
      result.current.prompt(pending);
    });
    await act(async () => {
      await result.current.reset();
    });

    expect(gateway.resetMemberBudgets).toHaveBeenCalledWith("team-999", ["u-1"]);
    expect(gateway.resetMemberBudgets).not.toHaveBeenCalledWith("team-123", expect.anything());
  });

  it("returns to prompting without its own toast when the team save fails", async () => {
    const gateway = buildGateway();
    gateway.saveTeam.mockRejectedValueOnce(new Error("team update failed"));
    const { result } = renderReset(gateway);

    await act(async () => {
      result.current.prompt(pendingFor(["u-1"]));
    });
    await act(async () => {
      await result.current.reset();
    });

    expect(result.current.state.phase).toBe("prompting");
    expect(gateway.resetMemberBudgets).not.toHaveBeenCalled();
    expect(gateway.refreshTeamData).not.toHaveBeenCalled();
    expect(toast.error).not.toHaveBeenCalled();
    expect(toast.fromError).not.toHaveBeenCalled();
    expect(toast.success).not.toHaveBeenCalled();
  });

  it("retries only the unsent members when a later batch fails, without re-saving the team", async () => {
    const userIds = Array.from({ length: MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES + 1 }, (_, i) => `u-${i}`);
    const gateway = buildGateway();
    gateway.resetMemberBudgets
      .mockResolvedValueOnce(
        userIds.slice(0, MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES).map((user_id) => ({ success: true, user_id })),
      )
      .mockRejectedValueOnce(new Error("second batch failed"));
    const { result } = renderReset(gateway);

    await act(async () => {
      result.current.prompt(pendingFor(userIds));
    });
    await act(async () => {
      await result.current.reset();
    });

    expect(result.current.state.phase).toBe("resetFailed");
    expect(toast.error).toHaveBeenCalledWith(
      `Reset ${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES} of ${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES + 1} member budgets; the rest could not be reset`,
    );

    gateway.resetMemberBudgets.mockResolvedValueOnce([
      { success: true, user_id: `u-${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES}` },
    ]);
    await act(async () => {
      await result.current.retry();
    });

    expect(gateway.saveTeam).toHaveBeenCalledTimes(1);
    expect(gateway.resetMemberBudgets).toHaveBeenCalledTimes(3);
    expect(gateway.resetMemberBudgets.mock.calls[2][0]).toBe("team-123");
    expect(gateway.resetMemberBudgets.mock.calls[2][1]).toEqual([`u-${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES}`]);
    expect(toast.success).toHaveBeenCalledWith(
      `Reset ${MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES + 1} member budgets to the team default`,
    );
    expect(result.current.state.phase).toBe("idle");
  });

  it("reports partial progress when the first batch fails", async () => {
    const gateway = buildGateway();
    gateway.resetMemberBudgets.mockRejectedValueOnce(new Error("batch failed"));
    const { result } = renderReset(gateway);

    await act(async () => {
      result.current.prompt(pendingFor(["u-1", "u-2"]));
    });
    await act(async () => {
      await result.current.reset();
    });

    expect(result.current.state.phase).toBe("resetFailed");
    expect(toast.fromError).toHaveBeenCalledWith("Team updated, but member budgets could not be reset");
    expect(gateway.refreshTeamData).toHaveBeenCalledTimes(1);
  });

  it("reports members the backend could not reset and closes", async () => {
    const gateway = buildGateway();
    gateway.resetMemberBudgets.mockResolvedValueOnce([
      { success: true, user_id: "u-1" },
      { success: false, user_id: "u-2" },
    ]);
    const { result } = renderReset(gateway);

    await act(async () => {
      result.current.prompt(pendingFor(["u-1", "u-2"]));
    });
    await act(async () => {
      await result.current.reset();
    });

    expect(toast.error).toHaveBeenCalledWith("Team updated, but 1 member budget could not be reset");
    expect(toast.success).not.toHaveBeenCalled();
    expect(result.current.state.phase).toBe("idle");
    expect(gateway.refreshTeamData).toHaveBeenCalledTimes(1);
  });

  it("saves the team once and closes on keep custom", async () => {
    const gateway = buildGateway();
    const { result } = renderReset(gateway);

    await act(async () => {
      result.current.prompt(pendingFor(["u-1"]));
    });
    await act(async () => {
      await result.current.keepCustom();
    });

    expect(gateway.saveTeam).toHaveBeenCalledTimes(1);
    expect(toast.success).toHaveBeenCalledWith("Team settings updated successfully");
    expect(gateway.resetMemberBudgets).not.toHaveBeenCalled();
    expect(gateway.refreshTeamData).toHaveBeenCalledTimes(1);
    expect(result.current.state.phase).toBe("idle");
  });

  it("abandons the pending update on dismiss", async () => {
    const gateway = buildGateway();
    const { result } = renderReset(gateway);

    await act(async () => {
      result.current.prompt(pendingFor(["u-1"]));
    });
    await act(async () => {
      result.current.dismiss();
    });

    expect(result.current.state.phase).toBe("idle");
    expect(gateway.saveTeam).not.toHaveBeenCalled();
    expect(gateway.resetMemberBudgets).not.toHaveBeenCalled();
  });

  it("skips the refresh and stays idle when dismissed while a reset is in flight", async () => {
    const gateway = buildGateway();
    const bulkDone = Promise.withResolvers<{ success: boolean; user_id: string }[]>();
    gateway.resetMemberBudgets.mockReturnValueOnce(bulkDone.promise);
    const { result } = renderReset(gateway);

    await act(async () => {
      result.current.prompt(pendingFor(["u-1"]));
    });
    let resetPromise = Promise.resolve();
    await act(async () => {
      resetPromise = result.current.reset();
    });
    expect(result.current.state.phase).toBe("resetting");

    act(() => {
      result.current.dismiss();
    });
    await act(async () => {
      bulkDone.resolve([{ success: true, user_id: "u-1" }]);
      await resetPromise;
    });

    expect(result.current.state.phase).toBe("idle");
    expect(gateway.refreshTeamData).not.toHaveBeenCalled();
    expect(toast.success).toHaveBeenCalledWith("Reset 1 member budget to the team default");
  });

  it("does not return to prompting when the team save fails after a dismiss", async () => {
    const gateway = buildGateway();
    const saveDone = Promise.withResolvers<void>();
    gateway.saveTeam.mockReturnValueOnce(saveDone.promise);
    const { result } = renderReset(gateway);

    await act(async () => {
      result.current.prompt(pendingFor(["u-1"]));
    });
    let resetPromise = Promise.resolve();
    await act(async () => {
      resetPromise = result.current.reset();
    });

    act(() => {
      result.current.dismiss();
    });
    await act(async () => {
      saveDone.reject(new Error("team update failed"));
      await resetPromise;
    });

    expect(result.current.state.phase).toBe("idle");
    expect(gateway.resetMemberBudgets).not.toHaveBeenCalled();
    expect(gateway.refreshTeamData).not.toHaveBeenCalled();
  });

  it("stays idle without a refresh when the reset fails after a dismiss", async () => {
    const gateway = buildGateway();
    const bulkDone = Promise.withResolvers<{ success: boolean; user_id: string }[]>();
    gateway.resetMemberBudgets.mockReturnValueOnce(bulkDone.promise);
    const { result } = renderReset(gateway);

    await act(async () => {
      result.current.prompt(pendingFor(["u-1"]));
    });
    let resetPromise = Promise.resolve();
    await act(async () => {
      resetPromise = result.current.reset();
    });

    act(() => {
      result.current.dismiss();
    });
    await act(async () => {
      bulkDone.reject(new Error("bulk failed"));
      await resetPromise;
    });

    expect(result.current.state.phase).toBe("idle");
    expect(gateway.refreshTeamData).not.toHaveBeenCalled();
  });

  it("skips the refresh when dismissed while keep-custom is saving", async () => {
    const gateway = buildGateway();
    const saveDone = Promise.withResolvers<void>();
    gateway.saveTeam.mockReturnValueOnce(saveDone.promise);
    const { result } = renderReset(gateway);

    await act(async () => {
      result.current.prompt(pendingFor(["u-1"]));
    });
    let keepPromise = Promise.resolve();
    await act(async () => {
      keepPromise = result.current.keepCustom();
    });

    act(() => {
      result.current.dismiss();
    });
    await act(async () => {
      saveDone.resolve();
      await keepPromise;
    });

    expect(result.current.state.phase).toBe("idle");
    expect(gateway.refreshTeamData).not.toHaveBeenCalled();
  });
});
