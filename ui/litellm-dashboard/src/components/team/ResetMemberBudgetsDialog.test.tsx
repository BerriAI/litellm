import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import ResetMemberBudgetsDialog from "./ResetMemberBudgetsDialog";

const baseProps = () => ({
  open: true,
  memberCount: 3,
  newBudget: 10,
  applying: false,
  onResetAll: vi.fn(),
  onKeepCustom: vi.fn(),
  onCancel: vi.fn(),
});

describe("ResetMemberBudgetsDialog", () => {
  it("names the affected members and the new default they would follow", () => {
    render(<ResetMemberBudgetsDialog {...baseProps()} />);

    expect(screen.getByText("Reset member budgets?")).toBeInTheDocument();
    expect(screen.getByText(/3 members have a custom budget/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reset all to $10" })).toBeInTheDocument();
  });

  it("switches to singular copy for one custom-budget member", () => {
    render(<ResetMemberBudgetsDialog {...baseProps()} memberCount={1} />);

    expect(screen.getByText(/1 member has a custom budget/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reset to $10" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Keep custom budget" })).toBeInTheDocument();
  });

  it("routes each choice to its own handler", async () => {
    const user = userEvent.setup({ delay: null });
    const props = baseProps();
    render(<ResetMemberBudgetsDialog {...props} />);

    await user.click(screen.getByRole("button", { name: "Reset all to $10" }));
    expect(props.onResetAll).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Keep custom budgets" }));
    expect(props.onKeepCustom).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(props.onCancel).toHaveBeenCalledTimes(1);
  });

  it("locks every button while the apply is in flight", () => {
    render(<ResetMemberBudgetsDialog {...baseProps()} applying />);

    expect(screen.getByRole("button", { name: "Reset all to $10" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Keep custom budgets" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
  });

  it("renders nothing while closed", () => {
    render(<ResetMemberBudgetsDialog {...baseProps()} open={false} />);

    expect(screen.queryByText("Reset member budgets?")).not.toBeInTheDocument();
  });
});
