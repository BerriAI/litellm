import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { components } from "@/lib/http/schema";

import EditBudgetModal from "./edit_budget_modal";
import { chooseSelectOption } from "../../../../../tests/test-utils";

const { updateMock } = vi.hoisted(() => ({ updateMock: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/budgets/useBudgets", () => ({
  useUpdateBudget: () => ({ mutateAsync: updateMock }),
}));

type BudgetItem = components["schemas"]["BudgetListItem"];

const EXISTING_BUDGET: BudgetItem = {
  budget_id: "budget-alpha",
  max_budget: 100,
  budget_duration: "7d",
  tpm_limit: 1000,
  rpm_limit: 10,
  soft_budget: 25,
  budget_reset_at: "2026-02-01T00:00:00Z",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-02T00:00:00Z",
};

const renderModal = () =>
  render(<EditBudgetModal isModalVisible={true} setIsModalVisible={vi.fn()} existingBudget={EXISTING_BUDGET} />);

const save = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: "Save" }));

describe("EditBudgetModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    updateMock.mockResolvedValue(undefined);
  });

  it("keeps the existing spend limit and reset period when only a rate limit changes", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "500.567" } });
    await save(user);

    await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(1));
    const expected = {
      budget_id: "budget-alpha",
      max_budget: 100,
      budget_duration: "7d",
      tpm_limit: 500.57,
      rpm_limit: 10,
    };
    expect(updateMock.mock.calls[0][0]).toEqual(expected);
  });

  it("submits a changed spend limit and reset period", async () => {
    const user = userEvent.setup();
    renderModal();

    const resetPeriod = screen.getByRole("combobox", { name: "Reset period" });
    expect(screen.getByLabelText(/^Spend limit/)).toHaveValue(100);
    expect(resetPeriod).toHaveTextContent("per week");
    fireEvent.change(screen.getByLabelText(/^Spend limit/), { target: { value: "42.567" } });
    await chooseSelectOption(user, resetPeriod, "per month");
    await save(user);

    await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(1));
    expect(updateMock.mock.calls[0][0]).toMatchObject({ max_budget: 42.57, budget_duration: "30d" });
  });

  it("does not let the budget ID be edited", () => {
    renderModal();

    expect(screen.getByLabelText("Budget ID")).toBeDisabled();
  });
});
