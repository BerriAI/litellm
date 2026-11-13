import { QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { budgetTestNetwork, jsonResponse } from "@/../tests/budgetTestNetwork";
import type { budgetItem } from "../../hooks/budgets/useBudgets";
import { chooseSelectOption } from "@/../tests/test-utils";
import EditBudgetModal from "./EditBudgetModal";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
vi.unmock("@/lib/toast");
afterEach(() => vi.unstubAllGlobals());

const shared: budgetItem = {
  budget_id: "shared",
  max_budget: 12.345,
  budget_duration: "7d",
  tpm_limit: 1000,
  rpm_limit: 10,
  tpd_limit: 50000,
  soft_budget: null,
  budget_reset_at: null,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-02T00:00:00Z",
};
const profile: budgetItem = {
  ...shared,
  budget_id: "profile",
  is_profile: true,
  revision: "original",
  cap_policy: "pause_until_reset",
  budget_limits: [
    { budget_duration: "1mo", max_budget: 20 },
    { budget_duration: "12mo", max_budget: 100 },
  ],
};

function setup(budget = shared, status = 200) {
  const network = budgetTestNetwork(() => jsonResponse({}, status));
  const tree = (row: budgetItem) => (
    <QueryClientProvider client={network.client}>
      <EditBudgetModal isModalVisible setIsModalVisible={() => {}} existingBudget={row} />
    </QueryClientProvider>
  );
  const view = render(tree(budget));
  return { ...network, refresh: (next: budgetItem) => view.rerender(tree(next)) };
}

describe("EditBudgetModal", () => {
  it("prefills shared limits and sends only the changed rate limit", async () => {
    const { requests } = setup();
    expect(screen.getByLabelText(/^Spend limit/)).toHaveValue(12.345);
    expect(screen.getByLabelText("Batch tokens per day")).toHaveValue(50000);
    expect(screen.getByLabelText("Budget ID")).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "500" } });
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(requests.find((request) => request.path === "/budget/update")?.body).toEqual({
        budget_id: "shared",
        tpm_limit: 500,
      }),
    );
  });

  it("keeps a dirty profile draft and its original revision across refresh", async () => {
    const { requests, refresh } = setup(profile);
    fireEvent.change(screen.getByLabelText("Monthly cap (USD)"), { target: { value: "0" } });
    refresh({ ...profile, revision: "newer", budget_limits: [{ budget_duration: "1mo", max_budget: 50 }] });
    expect(screen.getByLabelText("Monthly cap (USD)")).toHaveValue(0);
    expect(screen.getByLabelText("Annual cap (USD)")).toHaveValue(100);
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(requests.find((request) => request.path === "/budget/update")?.body).toEqual({
        budget_id: "profile",
        revision: "original",
        budget_limits: [
          { budget_duration: "1mo", max_budget: 0 },
          { budget_duration: "12mo", max_budget: 100 },
        ],
      }),
    );
  });

  it("changes guidance without resending caps and explains stale-profile rejection", async () => {
    const { requests } = setup(profile, 409);
    const user = userEvent.setup();
    await chooseSelectOption(
      user,
      screen.getByRole("combobox", { name: "At the cap" }),
      "Pause and allow an increase request",
    );
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("This profile changed since you loaded it.");
    expect(requests.find((request) => request.path === "/budget/update")?.body).toEqual({
      budget_id: "profile",
      revision: "original",
      cap_policy: "pause_allow_increase",
    });
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  });
});
