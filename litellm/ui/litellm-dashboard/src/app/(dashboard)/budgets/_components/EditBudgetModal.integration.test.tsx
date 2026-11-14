import { QueryClientProvider } from "@tanstack/react-query";
import { budgetTestNetwork } from "@/../tests/budgetTestNetwork";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { components } from "@/lib/http/schema";

import EditBudgetModal from "./EditBudgetModal";
import { chooseSelectOption } from "../../../../../tests/test-utils";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
vi.unmock("@/lib/toast");
afterEach(() => vi.unstubAllGlobals());
let network: ReturnType<typeof budgetTestNetwork>;

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

const renderModal = (budget = EXISTING_BUDGET) => {
  const tree = (row: BudgetItem) => (
    <QueryClientProvider client={network.client}>
      <EditBudgetModal isModalVisible={true} setIsModalVisible={vi.fn()} existingBudget={row} />
    </QueryClientProvider>
  );
  const view = render(tree(budget));
  return { refresh: (row: BudgetItem) => view.rerender(tree(row)) };
};

const save = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: "Save" }));

const resetPeriod = () => screen.getByRole("combobox", { name: "Reset period" });

describe("EditBudgetModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    network = budgetTestNetwork("/budget/update");
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

    await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
    expect(network.write.mock.calls[0][0]).toEqual({ budget_id: "budget-alpha", tpm_limit: 500 });
  });

  it("sends a later change after another field was changed and reverted", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "500" } });
    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "1000" } });
    fireEvent.change(screen.getByLabelText("Requests per minute"), { target: { value: "20" } });
    await save(user);

    await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
    expect(network.write.mock.calls[0][0]).toEqual({ budget_id: "budget-alpha", rpm_limit: 20 });
  });

  it("submits a changed spend limit and reset period", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText(/^Spend limit/), { target: { value: "42.567" } });
    await chooseSelectOption(user, resetPeriod(), "per month");
    await save(user);

    await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
    const expected = { budget_id: "budget-alpha", max_budget: 42.57, budget_duration: "30d" };
    expect(network.write.mock.calls[0][0]).toEqual(expected);
  });

  it("submits a cleared limit and an 'in total' reset period as null", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.clear(screen.getByLabelText("Tokens per minute"));
    await chooseSelectOption(user, resetPeriod(), "in total");
    await save(user);

    await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
    const expected = { budget_id: "budget-alpha", tpm_limit: null, budget_duration: null };
    expect(network.write.mock.calls[0][0]).toEqual(expected);
  });

  it("does not let the budget ID be edited", () => {
    renderModal();

    expect(screen.getByLabelText("Budget ID")).toBeDisabled();
  });

  it.each([
    { duration: "1mo", label: "per month", amount: 0 },
    { duration: "12mo", label: "per year", amount: 120 },
  ] as const)(
    "edits the existing $duration window without adding a scalar cap",
    async ({ duration, label, amount }) => {
      const user = userEvent.setup();
      const limit = { budget_duration: duration, max_budget: amount, reset_at: "2027-01-01T00:00:00Z" };
      const row: BudgetItem = {
        ...EXISTING_BUDGET,
        max_budget: null,
        budget_duration: null,
        budget_limits: [limit],
      };
      renderModal(row);
      expect(screen.getByLabelText(/^Spend limit/)).toHaveValue(amount);
      expect(resetPeriod()).toHaveTextContent(label);
      fireEvent.change(screen.getByLabelText(/^Spend limit/), { target: { value: "75" } });
      await save(user);
      await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
      expect(network.write.mock.calls[0][0]).toEqual({
        budget_id: "budget-alpha",
        budget_limits: [{ ...limit, max_budget: 75 }],
      });
    },
  );

  it.each([
    { duration: "1mo", label: "per month" },
    { duration: "12mo", label: "per year" },
  ] as const)("edits both stored caps when a scalar and $duration window coexist", async ({ duration, label }) => {
    const user = userEvent.setup();
    const limit = { budget_duration: duration, max_budget: 100, reset_at: "2027-01-01T00:00:00Z" };
    const row = { ...EXISTING_BUDGET, max_budget: 25, budget_duration: "24h", budget_limits: [limit] };
    renderModal(row);

    expect(screen.getByLabelText(/^Spend limit/)).toHaveValue(100);
    expect(resetPeriod()).toHaveTextContent(label);
    expect(screen.getByLabelText(/^Additional spend limit/)).toHaveValue(25);
    expect(screen.getByRole("combobox", { name: "Additional reset period" })).toHaveTextContent("per day");

    fireEvent.change(screen.getByLabelText(/^Spend limit/), { target: { value: "75" } });
    fireEvent.change(screen.getByLabelText(/^Additional spend limit/), { target: { value: "0" } });
    await save(user);
    await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
    expect(network.write.mock.calls[0][0]).toEqual({
      budget_id: "budget-alpha",
      max_budget: 0,
      budget_limits: [{ ...limit, max_budget: 75 }],
    });
  });

  it("shows all three caps and clears a zero scalar cap without resending calendar limits", async () => {
    const user = userEvent.setup();
    const row: BudgetItem = {
      ...EXISTING_BUDGET,
      max_budget: 0,
      budget_duration: "1h",
      budget_limits: [
        { budget_duration: "1mo", max_budget: 50 },
        { budget_duration: "12mo", max_budget: 500 },
      ],
    };
    renderModal(row);
    expect(screen.getByLabelText(/^Monthly limit/)).toHaveValue(50);
    expect(screen.getByLabelText("Annual limit (USD)")).toHaveValue(500);
    expect(screen.getByLabelText(/^Additional spend limit/)).toHaveValue(0);
    const scalarPeriod = screen.getByRole("combobox", { name: "Additional reset period" });
    expect(scalarPeriod).toHaveTextContent("hourly");

    fireEvent.change(screen.getByLabelText(/^Additional spend limit/), { target: { value: "" } });
    await chooseSelectOption(user, scalarPeriod, "in total");
    await save(user);
    await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
    expect(network.write.mock.calls[0][0]).toEqual({
      budget_id: "budget-alpha",
      max_budget: null,
      budget_duration: null,
    });
  });

  it("keeps a dirty combined budget on refresh and saves only the new annual cap", async () => {
    const user = userEvent.setup();
    const row: BudgetItem = {
      ...EXISTING_BUDGET,
      max_budget: null,
      budget_duration: null,
      budget_limits: [
        { budget_duration: "1mo", max_budget: 0 },
        { budget_duration: "12mo", max_budget: 100 },
      ],
    };
    const { refresh } = renderModal(row);
    expect(screen.getByLabelText(/^Monthly limit/)).toHaveValue(0);
    fireEvent.change(screen.getByLabelText("Annual limit (USD)"), { target: { value: "200" } });
    refresh({
      ...row,
      budget_limits: [
        { budget_duration: "1mo", max_budget: 80 },
        { budget_duration: "12mo", max_budget: 100 },
      ],
    });
    expect(screen.getByLabelText(/^Monthly limit/)).toHaveValue(0);
    expect(screen.getByLabelText("Annual limit (USD)")).toHaveValue(200);
    await save(user);
    await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
    expect(network.write.mock.calls[0][0]).toEqual({
      budget_id: "budget-alpha",
      budget_limits: [
        { budget_duration: "1mo", max_budget: 80 },
        { budget_duration: "12mo", max_budget: 200 },
      ],
    });
  });

  it("clears combined caps when returning to a single annual limit", async () => {
    const user = userEvent.setup();
    const row: BudgetItem = {
      ...EXISTING_BUDGET,
      max_budget: null,
      budget_duration: null,
      budget_limits: [
        { budget_duration: "1mo", max_budget: 10 },
        { budget_duration: "12mo", max_budget: 100 },
      ],
    };
    renderModal(row);
    await chooseSelectOption(user, resetPeriod(), "per year");
    expect(screen.getByLabelText(/^Spend limit/)).toHaveValue(100);
    fireEvent.change(screen.getByLabelText(/^Spend limit/), { target: { value: "0" } });
    await save(user);
    await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
    const expected = {
      budget_id: "budget-alpha",
      max_budget: 0,
      budget_duration: "12mo",
      budget_limits: null,
    };
    expect(network.write.mock.calls[0][0]).toEqual(expected);
  });

  it("keeps the monthly amount when switching from both to monthly without editing the amount", async () => {
    const user = userEvent.setup();
    const row: BudgetItem = {
      ...EXISTING_BUDGET,
      max_budget: null,
      budget_duration: null,
      budget_limits: [
        { budget_duration: "1mo", max_budget: 10 },
        { budget_duration: "12mo", max_budget: 100 },
      ],
    };
    renderModal(row);
    await chooseSelectOption(user, resetPeriod(), "per month");
    expect(screen.getByLabelText(/^Spend limit/)).toHaveValue(10);
    await save(user);
    await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
    const expected = {
      budget_id: "budget-alpha",
      max_budget: 10,
      budget_duration: "30d",
      budget_limits: null,
    };
    expect(network.write.mock.calls[0][0]).toEqual(expected);
  });

  it("keeps the annual cap when adding a monthly limit to an annual budget", async () => {
    const user = userEvent.setup();
    renderModal({ ...EXISTING_BUDGET, max_budget: 120, budget_duration: "12mo" });
    await chooseSelectOption(user, resetPeriod(), "per month and year");
    expect(screen.getByLabelText("Annual limit (USD)")).toHaveValue(120);
    fireEvent.change(screen.getByLabelText(/^Monthly limit/), { target: { value: "10" } });
    await save(user);
    await waitFor(() => expect(network.write).toHaveBeenCalledTimes(1));
    const expected = {
      budget_id: "budget-alpha",
      max_budget: null,
      budget_duration: null,
      budget_limits: [
        { budget_duration: "1mo", max_budget: 10 },
        { budget_duration: "12mo", max_budget: 120 },
      ],
    };
    expect(network.write.mock.calls[0][0]).toEqual(expected);
  });
});
