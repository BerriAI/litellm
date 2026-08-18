import React from "react";
import { describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { renderWithProviders } from "@/../tests/test-utils";
import EditUserModal from "./edit_user";

const POSSIBLE_UI_ROLES = {
  proxy_admin: { ui_label: "Admin", description: "Can create keys, teams, users" },
  internal_user: { ui_label: "Internal User", description: "Can create keys for themselves" },
};

const USER = {
  user_id: "user-123",
  user_email: "seed@example.com",
  user_role: "internal_user",
  spend: 3.5,
  max_budget: 10,
  budget_duration: "24h",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-02T00:00:00Z",
  teams: ["team-a"],
  models: ["gpt-4"],
  key_count: 7,
};

const renderModal = (overrides: Partial<React.ComponentProps<typeof EditUserModal>> = {}) => {
  const onSubmit = vi.fn();
  const onCancel = vi.fn();
  renderWithProviders(
    <EditUserModal
      visible
      possibleUIRoles={POSSIBLE_UI_ROLES}
      onCancel={onCancel}
      user={USER}
      onSubmit={onSubmit}
      {...overrides}
    />,
  );
  return { onSubmit, onCancel };
};

const save = async (user: ReturnType<typeof userEvent.setup>) => {
  const buttons = screen.getAllByRole("button", { name: "Save" });
  await user.click(buttons[0]);
};

describe("EditUserModal", () => {
  it("renders nothing when there is no user", () => {
    renderWithProviders(
      <EditUserModal visible possibleUIRoles={POSSIBLE_UI_ROLES} onCancel={vi.fn()} user={null} onSubmit={vi.fn()} />,
    );
    expect(screen.queryByText(/Edit User/)).not.toBeInTheDocument();
  });

  it("titles the modal with the user id", async () => {
    renderModal();
    expect(await screen.findByText("Edit User user-123")).toBeInTheDocument();
  });

  it("submits exactly the six bound fields, seeded from the user, and drops every other user key", async () => {
    const user = userEvent.setup();
    const { onSubmit, onCancel } = renderModal();
    await screen.findByText("Edit User user-123");

    await save(user);

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    const payload = onSubmit.mock.calls[0][0];
    expect(Object.keys(payload).sort()).toEqual([
      "budget_duration",
      "max_budget",
      "spend",
      "user_email",
      "user_id",
      "user_role",
    ]);
    const seededPayload = {
      user_id: "user-123",
      user_email: "seed@example.com",
      user_role: "internal_user",
      spend: 3.5,
      max_budget: 10,
      budget_duration: "24h",
    };
    expect(payload).toEqual(seededPayload);
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("submits the edited email as a string", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal();
    const email = await screen.findByLabelText("User Email");
    await user.clear(email);
    await user.type(email, "edited@example.com");

    await save(user);

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0].user_email).toBe("edited@example.com");
  });

  it("submits spend as a number and max_budget as a string once both are retyped", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal();
    const spend = await screen.findByLabelText("Spend (USD)");
    await user.clear(spend);
    await user.type(spend, "42.567");
    const maxBudget = screen.getByLabelText("User Budget (USD)");
    await user.clear(maxBudget);
    await user.type(maxBudget, "77.25");

    await save(user);

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    const payload = onSubmit.mock.calls[0][0];
    expect(payload.spend).toBe(42.567);
    expect(payload.max_budget).toBe("77.25");
  });

  it("keeps a cleared spend and a cleared max_budget distinguishable from zero", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal();
    const spend = await screen.findByLabelText("Spend (USD)");
    await user.clear(spend);
    const maxBudget = screen.getByLabelText("User Budget (USD)");
    await user.clear(maxBudget);

    await save(user);

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    const payload = onSubmit.mock.calls[0][0];
    expect(payload.spend).toBeNull();
    expect(payload.max_budget).toBe("");
  });

  it("clamps a negative spend up to the minimum on blur", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal();
    const spend = await screen.findByLabelText("Spend (USD)");
    await user.clear(spend);
    await user.type(spend, "-5");

    await save(user);

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0].spend).toBe(0);
  });

  it("blocks the whole submit while max_budget is below its minimum", async () => {
    const user = userEvent.setup();
    const { onSubmit, onCancel } = renderModal();
    const maxBudget = await screen.findByLabelText("User Budget (USD)");
    await user.clear(maxBudget);
    await user.type(maxBudget, "-5");

    await save(user);

    expect(onSubmit).not.toHaveBeenCalled();
    expect(onCancel).not.toHaveBeenCalled();
  });

  it("submits the selected role value, not its label", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    const { onSubmit } = renderModal();
    await screen.findByText("Edit User user-123");
    await user.click(screen.getByLabelText("User Role"));
    await user.click(await screen.findByTitle("Admin"));

    await save(user);

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0].user_role).toBe("proxy_admin");
  });

  it("submits the selected budget duration code", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    const { onSubmit } = renderModal();
    await screen.findByText("Edit User user-123");
    await user.click(screen.getByLabelText("Reset Budget"));
    await user.click(await screen.findByRole("option", { name: "weekly" }));

    await save(user);

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0].budget_duration).toBe("7d");
  });

  it("does not submit when the user cancels", async () => {
    const user = userEvent.setup();
    const { onSubmit, onCancel } = renderModal();
    await screen.findByText("Edit User user-123");
    await user.click(screen.getByRole("button", { name: /close/i }));
    expect(onSubmit).not.toHaveBeenCalled();
    await waitFor(() => expect(onCancel).toHaveBeenCalledTimes(1));
  });

  it("forwards null fields from the loaded user unchanged", async () => {
    const actor = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    const { onSubmit } = renderModal({ user: { ...USER, spend: null, max_budget: null, budget_duration: null } });

    await save(actor);

    const nulledPayload = {
      user_email: "seed@example.com",
      user_id: "user-123",
      user_role: "internal_user",
      spend: null,
      max_budget: null,
      budget_duration: null,
    };

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0]?.[0]).toEqual(nulledPayload);
  });
});
