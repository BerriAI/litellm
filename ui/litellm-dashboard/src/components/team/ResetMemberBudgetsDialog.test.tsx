import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import ResetMemberBudgetsDialog from "./ResetMemberBudgetsDialog";
import type { MemberBudgetResetState } from "./useMemberBudgetReset";

const pendingFor = (memberCount: number, newBudget = 10) => ({
  teamId: "team-123",
  updateData: {},
  userIds: Array.from({ length: memberCount }, (_, i) => `u-${i}`),
  newBudget,
});

const promptingWith = (memberCount: number, newBudget = 10): MemberBudgetResetState => ({
  phase: "prompting",
  pending: pendingFor(memberCount, newBudget),
});

const defaultHandlers = () => ({
  onReset: vi.fn(),
  onRetry: vi.fn(),
  onKeep: vi.fn(),
  onDismiss: vi.fn(),
});

describe("ResetMemberBudgetsDialog", () => {
  it("names the members and the new default so the prompt is self-explanatory", () => {
    render(<ResetMemberBudgetsDialog state={promptingWith(3)} {...defaultHandlers()} />);

    expect(
      screen.getByText(
        "3 members have a custom budget, so the new team default of $10.00 will not apply to them. " +
          "Reset them to the default, or keep the custom budgets?",
      ),
    ).toBeInTheDocument();
  });

  it("switches to singular for one member", () => {
    render(<ResetMemberBudgetsDialog state={promptingWith(1)} {...defaultHandlers()} />);

    expect(
      screen.getByText(
        "1 member has a custom budget, so the new team default of $10.00 will not apply to that member. " +
          "Reset it to the default, or keep the custom budget?",
      ),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reset to $10.00" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Keep custom budget" })).toBeInTheDocument();
  });

  it("formats large budgets like the rest of the team page", () => {
    render(<ResetMemberBudgetsDialog state={promptingWith(2, 1000)} {...defaultHandlers()} />);

    expect(screen.getByRole("button", { name: "Reset all to $1,000.00" })).toBeInTheDocument();
    expect(screen.getByText(/new team default of \$1,000\.00/)).toBeInTheDocument();
  });

  it("routes each choice to the matching action", async () => {
    const handlers = defaultHandlers();
    const user = userEvent.setup();
    render(<ResetMemberBudgetsDialog state={promptingWith(2)} {...handlers} />);

    await user.click(screen.getByRole("button", { name: "Reset all to $10.00" }));
    expect(handlers.onReset).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Keep custom budgets" }));
    expect(handlers.onKeep).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(handlers.onDismiss).toHaveBeenCalledTimes(1);
  });

  it("locks every button while the reset is in flight", () => {
    const state: MemberBudgetResetState = { phase: "resetting", pending: pendingFor(2), attempted: 0 };
    render(<ResetMemberBudgetsDialog state={state} {...defaultHandlers()} />);

    expect(screen.getByRole("button", { name: "Reset all to $10.00" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Keep custom budgets" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
  });

  it("offers retry or cancel after a failed reset", async () => {
    const handlers = defaultHandlers();
    const user = userEvent.setup();
    const state: MemberBudgetResetState = { phase: "resetFailed", pending: pendingFor(2), attempted: 0 };
    render(<ResetMemberBudgetsDialog state={state} {...handlers} />);

    await user.click(screen.getByRole("button", { name: "Retry reset" }));
    expect(handlers.onRetry).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(handlers.onDismiss).toHaveBeenCalledTimes(1);

    expect(screen.queryByRole("button", { name: "Keep custom budgets" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Reset all to $10.00" })).not.toBeInTheDocument();
  });

  it("renders nothing while idle", () => {
    render(<ResetMemberBudgetsDialog state={{ phase: "idle" }} {...defaultHandlers()} />);

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
