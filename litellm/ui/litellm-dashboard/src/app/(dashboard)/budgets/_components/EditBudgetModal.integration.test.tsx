import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { components } from "@/lib/http/schema";

import EditBudgetModal from "./EditBudgetModal";
import { chooseSelectOption } from "../../../../../tests/test-utils";

const { updateMock } = vi.hoisted(() => ({ updateMock: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/budgets/useBudgets", () => ({
  useUpdateBudget: () => ({ mutateAsync: updateMock }),
}));

type BudgetItem = components["schemas"]["BudgetListItem"];

const EXISTING_BUDGET: BudgetItem = {
  budget_id: "budget-alpha",
  max_budget: 12.345,
  budget_duration: "7d",
  tpm_limit: 1000,
  rpm_limit: 10,
  tpd_limit: 50000,
  soft_budget: 25,
  budget_reset_at: "2026-02-01T00:00:00Z",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-02T00:00:00Z",
};

const renderModal = () =>
  render(<EditBudgetModal isModalVisible={true} setIsModalVisible={vi.fn()} existingBudget={EXISTING_BUDGET} />);

const save = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: "Save" }));

const resetPeriod = () => screen.getByRole("combobox", { name: "Reset period" });

describe("EditBudgetModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    updateMock.mockResolvedValue(undefined);
  });

  it("prefills every field from the existing budget", () => {
    renderModal();

    expect(screen.getByLabelText(/^Spend limit/)).toHaveValue(12.345);
    expect(resetPeriod()).toHaveTextContent("per week");
    expect(screen.getByLabelText("Tokens per minute")).toHaveValue(1000);
    expect(screen.getByLabelText("Requests per minute")).toHaveValue(10);
    expect(screen.getByLabelText("Batch tokens per day")).toHaveValue(50000);
  });

  it("sends only the changed rate limit so the stored spend limit and reset period are left as they are", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "500" } });
    await save(user);

    await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(1));
    expect(updateMock.mock.calls[0][0]).toEqual({ budget_id: "budget-alpha", tpm_limit: 500 });
  });

  it("sends a later change after another field was changed and reverted", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "500" } });
    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "1000" } });
    fireEvent.change(screen.getByLabelText("Requests per minute"), { target: { value: "20" } });
    await save(user);

    await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(1));
    expect(updateMock.mock.calls[0][0]).toEqual({ budget_id: "budget-alpha", rpm_limit: 20 });
  });

  it("submits a changed spend limit and reset period", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText(/^Spend limit/), { target: { value: "42.567" } });
    await chooseSelectOption(user, resetPeriod(), "per month");
    await save(user);

    await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(1));
    const expected = { budget_id: "budget-alpha", max_budget: 42.57, budget_duration: "30d" };
    expect(updateMock.mock.calls[0][0]).toEqual(expected);
  });

  it("submits a cleared limit and an 'in total' reset period as null", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.clear(screen.getByLabelText("Tokens per minute"));
    await chooseSelectOption(user, resetPeriod(), "in total");
    await save(user);

    await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(1));
    const expected = { budget_id: "budget-alpha", tpm_limit: null, budget_duration: null };
    expect(updateMock.mock.calls[0][0]).toEqual(expected);
  });

  it("does not let the budget ID be edited", () => {
    renderModal();

    expect(screen.getByLabelText("Budget ID")).toBeDisabled();
  });
});
