import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import MemberModal from "./EditMembership";

const windows = [
  { budget_duration: "24h", max_budget: 20 },
  { budget_duration: "30d", max_budget: 200 },
];

function renderMember(budget_limits = windows, defaultBudgetWindows = windows) {
  const onSubmit = vi.fn();
  renderWithProviders(
    <MemberModal
      visible
      mode="edit"
      initialData={{ user_id: "member", role: "user", budget_limits, max_budget_in_team: 50 }}
      onCancel={vi.fn()}
      onSubmit={onSubmit}
      config={{
        title: "Edit Member",
        roleOptions: [{ label: "User", value: "user" }],
        showUserId: true,
        showBudgetWindows: true,
        defaultBudgetWindows,
        additionalFields: [{ name: "max_budget_in_team", label: "Single budget", type: "numerical" }],
      }}
    />,
  );
  return onSubmit;
}

describe("member budget windows form", () => {
  it("should render both configured windows", () => {
    renderMember();
    expect(screen.getByRole("spinbutton", { name: "Window 1 budget (USD)" })).toHaveValue(20);
    expect(screen.getByRole("spinbutton", { name: "Window 2 budget (USD)" })).toHaveValue(200);
    expect(screen.getByRole("switch", { name: "Custom budget windows" })).toBeChecked();
  });

  it("should keep windows off and save an unrelated member edit without requiring a budget", async () => {
    const onSubmit = renderMember([], []);
    expect(screen.getByRole("switch", { name: "Custom budget windows" })).not.toBeChecked();
    expect(screen.getByRole("img", { name: /20 dollar daily cap/ })).toBeInTheDocument();
    expect(screen.queryByRole("spinbutton", { name: "Window 1 budget (USD)" })).not.toBeInTheDocument();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Save Changes" }));
    });
    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ budget_limits: [] })));
  });
  it("should inherit team defaults, then save a custom zero-dollar cap without changing the single budget", async () => {
    const onSubmit = renderMember([]);
    expect(screen.getByText("$20 daily, $200 monthly")).toBeInTheDocument();
    expect(screen.queryByRole("spinbutton", { name: "Window 1 budget (USD)" })).not.toBeInTheDocument();
    await act(async () => {
      fireEvent.click(screen.getByRole("switch", { name: "Custom budget windows" }));
    });
    await act(async () => {
      fireEvent.change(screen.getByRole("spinbutton", { name: "Window 1 budget (USD)" }), { target: { value: "0" } });
      fireEvent.click(screen.getByRole("button", { name: "Save Changes" }));
    });
    const expected = {
      user_id: "member",
      role: "user",
      max_budget_in_team: 50,
      budget_limits: [{ ...windows[0], max_budget: 0 }, windows[1]],
    };
    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith(expected));
  });

  it("should send an empty override to restore inheritance", async () => {
    const onSubmit = renderMember();
    await act(async () => {
      fireEvent.click(screen.getByRole("switch", { name: "Custom budget windows" }));
      fireEvent.click(screen.getByRole("button", { name: "Save Changes" }));
    });
    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ budget_limits: [] })));
  });

  it("should explain an incomplete custom window and prevent saving it", async () => {
    const onSubmit = renderMember([], []);
    await act(async () => {
      fireEvent.click(screen.getByRole("switch", { name: "Custom budget windows" }));
      fireEvent.click(screen.getByRole("button", { name: "Save Changes" }));
    });
    expect(await screen.findByText("Enter a budget")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("should explain duplicate durations and prevent saving them", async () => {
    const onSubmit = renderMember([windows[0], windows[0]]);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Save Changes" }));
    });
    expect(await screen.findByText("Use each budget duration only once")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
