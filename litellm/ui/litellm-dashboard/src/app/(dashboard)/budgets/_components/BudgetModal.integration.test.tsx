import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import BudgetModal from "./BudgetModal";
import { chooseSelectOption } from "../../../../../tests/test-utils";

const { createMock } = vi.hoisted(() => ({ createMock: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/budgets/useBudgets", () => ({
  useCreateBudget: () => ({ mutateAsync: createMock }),
}));

const renderModal = () => render(<BudgetModal isModalVisible={true} setIsModalVisible={vi.fn()} />);

const create = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: "Create budget" }));

const resetPeriod = () => screen.getByRole("combobox", { name: "Reset period" });

describe("BudgetModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    createMock.mockResolvedValue(undefined);
  });

  it("submits the spend limit and reset period alongside the rate limits", async () => {
    const user = userEvent.setup();
    renderModal();

    expect(screen.getByLabelText(/^Spend limit/)).toBeVisible();
    expect(resetPeriod()).toBeVisible();
    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "budget-alpha" } });
    fireEvent.change(screen.getByLabelText(/^Spend limit/), { target: { value: "42.567" } });
    await chooseSelectOption(user, resetPeriod(), "per month");
    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "500" } });
    fireEvent.change(screen.getByLabelText("Requests per minute"), { target: { value: "7" } });
    fireEvent.change(screen.getByLabelText("Batch tokens per day"), { target: { value: "90000" } });
    await create(user);

    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
    const expected = {
      budget_id: "budget-alpha",
      max_budget: 42.57,
      budget_duration: "30d",
      tpm_limit: 500,
      rpm_limit: 7,
      tpd_limit: 90000,
    };
    expect(createMock.mock.calls[0][0]).toEqual(expected);
  });

  it("submits a spend limit that never resets as a null reset period", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "budget-alpha" } });
    fireEvent.change(screen.getByLabelText(/^Spend limit/), { target: { value: "10" } });
    expect(resetPeriod()).toHaveTextContent("in total");
    await chooseSelectOption(user, resetPeriod(), "per week");
    await chooseSelectOption(user, resetPeriod(), "in total");
    await create(user);

    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
    expect(createMock.mock.calls[0][0]).toEqual({ budget_id: "budget-alpha", max_budget: 10, budget_duration: null });
  });

  it("submits a cleared number field as null", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "budget-alpha" } });
    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "5" } });
    await user.clear(screen.getByLabelText("Tokens per minute"));
    await create(user);

    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
    expect(createMock.mock.calls[0][0]).toEqual({ budget_id: "budget-alpha", tpm_limit: null });
  });

  it("blocks submit while Budget ID is empty", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "5" } });
    await create(user);

    await waitFor(() => expect(screen.getByLabelText("Budget ID")).toHaveAttribute("aria-invalid", "true"));
    expect(createMock).not.toHaveBeenCalled();
  });

  it.each([
    ["Pause and allow temporary increase requests", "pause_allow_increase"],
    ["Pause until reset", "pause_until_reset"],
  ])("creates a profile with independent monthly and annual limits and %s", async (label, policy) => {
    const user = userEvent.setup();
    renderModal();
    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "standard-profile" } });
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Budget type" }), "Reusable profile");
    fireEvent.change(screen.getByLabelText("Monthly limit per person (USD)"), { target: { value: "100" } });
    fireEvent.change(screen.getByLabelText("Annual limit per person (USD)"), { target: { value: "1000" } });
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "At the cap" }), label);
    await create(user);
    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
    const expected = {
      budget_id: "standard-profile",
      is_profile: true,
      budget_limits: [
        { budget_duration: "1mo", max_budget: 100 },
        { budget_duration: "12mo", max_budget: 1000 },
      ],
      cap_policy: policy,
      max_budget: null,
      budget_duration: null,
      tpm_limit: null,
      rpm_limit: null,
      tpd_limit: null,
    };
    expect(createMock.mock.calls[0][0]).toEqual(expected);
  });

  it("requires a limit for a profile while allowing a zero-dollar cap", async () => {
    const user = userEvent.setup();
    renderModal();
    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "paused-profile" } });
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Budget type" }), "Reusable profile");
    await create(user);
    expect(await screen.findByText("Set a monthly or annual limit")).toBeVisible();
    expect(createMock).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Monthly limit per person (USD)"), { target: { value: "0" } });
    await create(user);
    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
    expect(createMock.mock.calls[0][0].budget_limits).toEqual([{ budget_duration: "1mo", max_budget: 0 }]);
  });
});
