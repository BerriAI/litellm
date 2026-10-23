import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { toast } from "@/lib/toast";
import type { MemberBudgetBackfillResult, MemberBudgetResetPending } from "./memberBudgetReset";
import { backfillToastMessage, useMemberBudgetReset, type MemberBudgetResetGateway } from "./useMemberBudgetReset";

const pending: MemberBudgetResetPending = {
  updateData: { team_id: "team-123", team_member_budget: 20 },
  newBudget: 20,
  preview: { below: 2, above: 1, otherResetPeriod: 0 },
};

const buildGateway = (result?: MemberBudgetBackfillResult) => ({
  saveTeam: vi.fn(async (_updateData: Record<string, unknown>) => result),
  refreshTeamData: vi.fn(async () => {}),
});

const renderReset = (gateway: MemberBudgetResetGateway) => renderHook(() => useMemberBudgetReset(gateway));

const promptedReset = async (gateway: MemberBudgetResetGateway) => {
  const rendered = renderReset(gateway);
  await act(async () => {
    rendered.result.current.prompt(pending);
  });
  return rendered;
};

describe("backfillToastMessage", () => {
  it.each([
    [undefined, "Team settings updated successfully"],
    [{ moved_to_default: 1, skipped_other_reset_period: 0 }, "Moved 1 member to the $20.00 default"],
    [
      { moved_to_default: 3, skipped_other_reset_period: 1 },
      "Moved 3 members to the $20.00 default. 1 member resets on a different schedule and was not changed",
    ],
    [
      { moved_to_default: 0, skipped_other_reset_period: 2 },
      "Moved 0 members to the $20.00 default. 2 members reset on a different schedule and were not changed",
    ],
  ])("reports %j as %s", (result, message) => {
    expect(backfillToastMessage(result, 20)).toBe(message);
  });
});

describe("useMemberBudgetReset", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("holds the update until a choice is saved", async () => {
    const gateway = buildGateway();
    const { result } = await promptedReset(gateway);

    expect(result.current.state).toEqual({ phase: "prompting", pending });
    expect(gateway.saveTeam).not.toHaveBeenCalled();
  });

  it("saves the update and the chosen direction in one request and reports the server's counts", async () => {
    const gateway = buildGateway({ moved_to_default: 2, skipped_other_reset_period: 0 });
    const { result } = await promptedReset(gateway);

    await act(async () => {
      await result.current.save("raise_to_default");
    });

    expect(gateway.saveTeam).toHaveBeenCalledTimes(1);
    expect(gateway.saveTeam).toHaveBeenCalledWith({
      team_id: "team-123",
      team_member_budget: 20,
      team_member_budget_backfill: "raise_to_default",
    });
    expect(toast.success).toHaveBeenCalledWith("Moved 2 members to the $20.00 default");
    expect(result.current.state.phase).toBe("idle");
    expect(gateway.refreshTeamData).toHaveBeenCalledTimes(1);
  });

  it("confirms a plain save when custom budgets are kept", async () => {
    const gateway = buildGateway({ moved_to_default: 0, skipped_other_reset_period: 0 });
    const { result } = await promptedReset(gateway);

    await act(async () => {
      await result.current.save("none");
    });

    expect(gateway.saveTeam).toHaveBeenCalledWith({ ...pending.updateData, team_member_budget_backfill: "none" });
    expect(toast.success).toHaveBeenCalledWith("Team settings updated successfully");
  });

  it("returns to the prompt when the save fails", async () => {
    const gateway = buildGateway();
    gateway.saveTeam.mockRejectedValueOnce(new Error("save failed"));
    const { result } = await promptedReset(gateway);

    await act(async () => {
      await result.current.save("lower_to_default");
    });

    expect(result.current.state).toEqual({ phase: "prompting", pending });
    expect(toast.success).not.toHaveBeenCalled();
    expect(gateway.refreshTeamData).not.toHaveBeenCalled();
  });

  it("drops the outcome of a save that was dismissed while in flight", async () => {
    const save = Promise.withResolvers<MemberBudgetBackfillResult | undefined>();
    const gateway = buildGateway();
    gateway.saveTeam.mockReturnValueOnce(save.promise);
    const { result } = await promptedReset(gateway);

    let saving: Promise<void> = Promise.resolve();
    act(() => {
      saving = result.current.save("all_to_default");
    });
    expect(result.current.state.phase).toBe("saving");
    act(() => {
      result.current.dismiss();
    });
    await act(async () => {
      save.resolve({ moved_to_default: 3, skipped_other_reset_period: 0 });
      await saving;
    });

    expect(result.current.state.phase).toBe("idle");
    expect(toast.success).not.toHaveBeenCalled();
    expect(gateway.refreshTeamData).not.toHaveBeenCalled();
  });
});
