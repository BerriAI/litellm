import { QueryClientProvider } from "@tanstack/react-query";
import { budgetTestNetwork } from "@/../tests/budgetTestNetwork";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import BudgetModal from "./BudgetModal";
import { chooseSelectOption } from "../../../../../tests/test-utils";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
vi.unmock("@/lib/toast");
afterEach(() => vi.unstubAllGlobals());
let network: ReturnType<typeof budgetTestNetwork>;

const renderModal = () =>
  render(
    <QueryClientProvider client={network.client}>
      <BudgetModal isModalVisible={true} setIsModalVisible={vi.fn()} />
    </QueryClientProvider>,
  );

const create = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: "Create budget" }));

const resetPeriod = () => screen.getByRole("combobox", { name: "Reset period" });

describe("BudgetModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    network = budgetTestNetwork("/budget/new");
  });

  it.each([
    ["per month", "30d"],
    ["per year", "12mo"],
  ])("submits the spend limit %s alongside the rate limits", async (label, duration) => {
    const user = userEvent.setup();
    renderModal();

    expect(screen.getByLabelText(/^Spend limit/)).toBeVisible();
    expect(resetPeriod()).toBeVisible();
    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "budget-alpha" } });
    fireEvent.change(screen.getByLabelText(/^Spend limit/), { target: { value: "42.567" } });
    await chooseSelectOption(user, resetPeriod(), label);
    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "500" } });
    fireEvent.change(screen.getByLabelText("Requests per minute"), { target: { value: "7" } });
    fireEvent.change(screen.getByLabelText("Batch tokens per day"), { target: { value: "90000" } });
    await create(user);

    await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
    const expected = {
      budget_id: "budget-alpha",
      max_budget: 42.57,
      budget_duration: duration,
      tpm_limit: 500,
      rpm_limit: 7,
      tpd_limit: 90000,
    };
    expect(network.write.mock.calls[0][0]).toEqual(expected);
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

    await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
    expect(network.write.mock.calls[0][0]).toEqual({
      budget_id: "budget-alpha",
      max_budget: 10,
      budget_duration: null,
    });
  });

  it("submits a cleared number field as null", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "budget-alpha" } });
    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "5" } });
    await user.clear(screen.getByLabelText("Tokens per minute"));
    await create(user);

    await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
    expect(network.write.mock.calls[0][0]).toEqual({ budget_id: "budget-alpha", tpm_limit: null });
  });

  it("blocks submit while Budget ID is empty", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "5" } });
    await create(user);

    await waitFor(() => expect(screen.getByLabelText("Budget ID")).toHaveAttribute("aria-invalid", "true"));
    expect(network.write).not.toHaveBeenCalled();
  });

  it("creates simultaneous monthly and annual limits through the existing reset selector", async () => {
    const user = userEvent.setup();
    renderModal();
    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "dual" } });
    await chooseSelectOption(user, resetPeriod(), "per month and year");
    fireEvent.change(screen.getByLabelText(/^Monthly limit/), { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText("Annual limit (USD)"), { target: { value: "120" } });
    await create(user);
    await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
    expect(network.write.mock.calls[0][0]).toEqual({
      budget_id: "dual",
      budget_limits: [
        { budget_duration: "1mo", max_budget: 0 },
        { budget_duration: "12mo", max_budget: 120 },
      ],
    });
  });

  it("requires both non-negative limits when the combined period is selected", async () => {
    const user = userEvent.setup();
    renderModal();
    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "dual" } });
    await chooseSelectOption(user, resetPeriod(), "per month and year");
    fireEvent.change(screen.getByLabelText(/^Monthly limit/), { target: { value: "-1" } });
    await create(user);
    await waitFor(() => expect(screen.getByLabelText(/^Monthly limit/)).toHaveAttribute("aria-invalid", "true"));
    expect(screen.getByLabelText("Annual limit (USD)")).toHaveAttribute("aria-invalid", "true");
    expect(network.write).not.toHaveBeenCalled();
  });
});
