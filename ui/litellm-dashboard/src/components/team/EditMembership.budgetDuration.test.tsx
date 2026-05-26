/**
 * LIT-2651: when a team member is edited, the Edit Member modal MUST surface a
 * Budget Reset Period selector and pass the selected value through onSubmit
 * (with the "" sentinel translated to `null` once it reaches teamMemberUpdateCall).
 *
 * Before this fix, the modal had no budget_duration field at all, so
 * /team/member_update created member budgets with budget_duration=null — and
 * the daily ResetBudgetJob never picks those up, so the per-member cap never
 * resets. Customer report: Telia, Cao Jie.
 */
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import EditMembership from "./EditMembership";

const editMemberConfig = {
  title: "Edit Member",
  showEmail: true,
  showUserId: true,
  roleOptions: [
    { label: "Admin", value: "admin" },
    { label: "User", value: "user" },
  ],
  // Match the shape declared in TeamInfo.tsx, including the new
  // Budget Reset Period field.
  additionalFields: [
    {
      name: "max_budget_in_team",
      label: <span>Team Member Budget (USD)</span>,
      type: "numerical" as const,
      step: 0.01,
      min: 0,
      placeholder: "Budget limit for this member within this team",
    },
    {
      name: "budget_duration",
      label: <span>Budget Reset Period</span>,
      type: "select" as const,
      placeholder: "No reset (lifetime cap)",
      options: [
        { label: "No reset (lifetime cap)", value: "" },
        { label: "Daily (24h)", value: "24h" },
        { label: "Weekly (7d)", value: "7d" },
        { label: "Monthly (30d)", value: "30d" },
      ],
    },
  ],
};

describe("EditMembership Budget Reset Period (LIT-2651)", () => {
  it("renders the Budget Reset Period field on the Edit Member modal", () => {
    renderWithProviders(
      <EditMembership
        visible={true}
        onCancel={vi.fn()}
        onSubmit={vi.fn()}
        mode="edit"
        initialData={{
          role: "user",
          user_id: "user-A",
          user_email: "alice@telia.example",
        }}
        config={editMemberConfig}
      />,
    );

    // The label is the regression assertion the customer reported: the field
    // simply did not exist on the modal.
    expect(screen.getByText("Budget Reset Period")).toBeInTheDocument();
  });

  it("pre-fills the field from initialData.budget_duration so existing members keep their period", async () => {
    renderWithProviders(
      <EditMembership
        visible={true}
        onCancel={vi.fn()}
        onSubmit={vi.fn()}
        mode="edit"
        initialData={{
          role: "user",
          user_id: "user-A",
          user_email: "alice@telia.example",
          // Existing period on the member's budget row.
          budget_duration: "7d",
        }}
        config={editMemberConfig}
      />,
    );

    // antd Select renders the current value as the visible item text. Use
    // findByText to wait for the form's setFieldsValue effect.
    expect(await screen.findByText("Weekly (7d)")).toBeInTheDocument();
  });

  it("submits the selected budget_duration through onSubmit", async () => {
    const onSubmit = vi.fn();
    renderWithProviders(
      <EditMembership
        visible={true}
        onCancel={vi.fn()}
        onSubmit={onSubmit}
        mode="edit"
        initialData={{
          role: "user",
          user_id: "user-A",
          user_email: "alice@telia.example",
          budget_duration: "7d",
        }}
        config={editMemberConfig}
      />,
    );

    // Submit the prefilled form — proves the prefilled value passes through
    // onSubmit, which is the contract handleMemberUpdate relies on.
    const submitButton = screen.getByRole("button", { name: "Save Changes" });
    act(() => {
      fireEvent.click(submitButton);
    });

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalled();
    });

    const submitted = onSubmit.mock.calls[0][0];
    expect(submitted.budget_duration).toBe("7d");
  });
});
