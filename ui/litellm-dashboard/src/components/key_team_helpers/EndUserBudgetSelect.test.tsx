import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { chooseSelectOption } from "../../../tests/test-utils";
import { EndUserBudgetSelect } from "./EndUserBudgetSelect";

const useBudgetOptions = vi.fn();
vi.mock("@/app/(dashboard)/hooks/budgets/useBudgetOptions", () => ({
  useBudgetOptions: (...args: unknown[]) => useBudgetOptions(...args),
}));

const BUDGETS = [
  { budget_id: "svc-a-budget", max_budget: 0.5, budget_duration: "30d", created_at: "", updated_at: "" },
  { budget_id: "svc-b-budget", max_budget: null, budget_duration: null, created_at: "", updated_at: "" },
];

describe("EndUserBudgetSelect", () => {
  it("lets an admin pick one of the proxy's budgets and reports its id", async () => {
    useBudgetOptions.mockReturnValue({ data: BUDGETS });
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<EndUserBudgetSelect accessToken="tok" value={null} onChange={onChange} canEdit />);

    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Default Customer Budget" }), /svc-a-budget/);

    expect(onChange).toHaveBeenLastCalledWith("svc-a-budget");
    expect(useBudgetOptions).toHaveBeenCalledWith("tok", true);
  });

  it("shows a budget's cap and reset window next to its id", async () => {
    useBudgetOptions.mockReturnValue({ data: BUDGETS });
    const user = userEvent.setup();
    render(<EndUserBudgetSelect accessToken="tok" value={null} onChange={vi.fn()} canEdit />);

    await user.click(screen.getByRole("combobox"));

    expect(await screen.findByRole("option", { name: /svc-a-budget/ })).toHaveTextContent("$0.5, resets 30d");
  });

  it("clears to null so the edit form can send an explicit empty value", async () => {
    useBudgetOptions.mockReturnValue({ data: BUDGETS });
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<EndUserBudgetSelect accessToken="tok" value="svc-a-budget" onChange={onChange} canEdit />);

    await user.click(screen.getByRole("button", { name: "Clear" }));

    expect(onChange).toHaveBeenLastCalledWith(null);
  });

  it("keeps the stored budget visible but read-only for a user who cannot change it", () => {
    useBudgetOptions.mockReturnValue({ data: undefined });
    render(<EndUserBudgetSelect accessToken="tok" value="svc-a-budget" onChange={vi.fn()} canEdit={false} />);

    const combobox = screen.getByRole("combobox", { name: "Default Customer Budget" });
    expect(combobox).toHaveValue("svc-a-budget");
    expect(combobox).toBeDisabled();
    expect(useBudgetOptions).toHaveBeenLastCalledWith("tok", false);
  });
});
