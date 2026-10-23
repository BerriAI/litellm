import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { MemberBudgetBackfillPreview } from "./memberBudgetReset";
import ResetMemberBudgetsDialog, { backfillOptions } from "./ResetMemberBudgetsDialog";
import type { MemberBudgetResetState } from "./useMemberBudgetReset";

const preview = (below: number, above: number, otherResetPeriod = 0): MemberBudgetBackfillPreview => ({
  below,
  above,
  otherResetPeriod,
});

const state = (phase: "prompting" | "saving", shown: MemberBudgetBackfillPreview): MemberBudgetResetState => ({
  phase,
  pending: { updateData: { team_id: "team-123" }, newBudget: 1250, preview: shown },
});

const renderDialog = (dialogState: MemberBudgetResetState) => {
  const handlers = { onSave: vi.fn(), onDismiss: vi.fn() };
  render(<ResetMemberBudgetsDialog state={dialogState} {...handlers} />);
  return handlers;
};

describe("backfillOptions", () => {
  it("offers every direction when members sit on both sides of the default", () => {
    expect(backfillOptions(preview(2, 1), 20)).toEqual([
      { value: "none", label: "Keep all custom budgets" },
      { value: "raise_to_default", label: "Raise 2 members below $20.00" },
      { value: "lower_to_default", label: "Lower 1 member above $20.00" },
      { value: "all_to_default", label: "Raise and lower all 3 members" },
    ]);
  });

  it.each([
    [preview(1, 0), ["none", "raise_to_default"]],
    [preview(0, 4), ["none", "lower_to_default"]],
  ])("hides directions that would move nobody", (shown, values) => {
    expect(backfillOptions(shown, 20).map((option) => option.value)).toEqual(values);
  });
});

describe("ResetMemberBudgetsDialog", () => {
  it("keeps custom budgets unless another direction is picked", async () => {
    const user = userEvent.setup();
    const { onSave } = renderDialog(state("prompting", preview(2, 1)));

    expect(screen.getByText("Apply the $1,250.00 default to custom budgets?")).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Keep all custom budgets" })).toBeChecked();
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(onSave).toHaveBeenCalledWith("none");
  });

  it("saves the picked direction", async () => {
    const user = userEvent.setup();
    const { onSave } = renderDialog(state("prompting", preview(2, 1)));

    await user.click(screen.getByRole("radio", { name: "Raise and lower all 3 members" }));
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(onSave).toHaveBeenCalledWith("all_to_default");
  });

  it("says how many members reset on another schedule", () => {
    renderDialog(state("prompting", preview(1, 0, 2)));

    expect(screen.getByText("2 members reset on a different schedule and won't change")).toBeInTheDocument();
  });

  it("cancels without saving", async () => {
    const user = userEvent.setup();
    const { onSave, onDismiss } = renderDialog(state("prompting", preview(1, 0)));

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onDismiss).toHaveBeenCalledTimes(1);
    expect(onSave).not.toHaveBeenCalled();
  });

  it("locks every control while the save is in flight", () => {
    renderDialog(state("saving", preview(1, 0)));

    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    expect(screen.getByRole("radio", { name: "Raise 1 member below $1,250.00" })).toHaveAttribute(
      "aria-disabled",
      "true",
    );
  });
});
