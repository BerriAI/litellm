import { QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { budgetTestNetwork, jsonResponse } from "@/../tests/budgetTestNetwork";
import { chooseSelectOption } from "@/../tests/test-utils";
import BudgetModal from "./BudgetModal";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
vi.unmock("@/lib/toast");
afterEach(() => vi.unstubAllGlobals());

function setup() {
  const network = budgetTestNetwork(() => jsonResponse({}));
  render(
    <QueryClientProvider client={network.client}>
      <BudgetModal isModalVisible setIsModalVisible={() => {}} />
    </QueryClientProvider>,
  );
  return network;
}

describe("BudgetModal", () => {
  it("creates a profile with independent zero monthly and positive annual caps", async () => {
    const { requests } = setup();
    const user = userEvent.setup();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Budget type" }), "Per-person profile");
    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "starter" } });
    fireEvent.change(screen.getByLabelText("Monthly cap (USD)"), { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText("Annual cap (USD)"), { target: { value: "75" } });
    await chooseSelectOption(
      user,
      screen.getByRole("combobox", { name: "At the cap" }),
      "Pause and allow an increase request",
    );
    await user.click(screen.getByRole("button", { name: "Create budget" }));
    const expected = {
      budget_id: "starter",
      is_profile: true,
      cap_policy: "pause_allow_increase",
      budget_limits: [
        { budget_duration: "1mo", max_budget: 0 },
        { budget_duration: "12mo", max_budget: 75 },
      ],
    };
    await waitFor(() => expect(requests.find((request) => request.path === "/budget/new")?.body).toEqual(expected));
  });

  it("keeps the existing shared-budget request and precision", async () => {
    const { requests } = setup();
    const user = userEvent.setup();
    fireEvent.change(screen.getByLabelText("Budget ID"), { target: { value: "shared" } });
    fireEvent.change(screen.getByLabelText(/^Spend limit/), { target: { value: "42.567" } });
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Reset period" }), "per month");
    fireEvent.change(screen.getByLabelText("Tokens per minute"), { target: { value: "500" } });
    await user.click(screen.getByRole("button", { name: "Create budget" }));
    const expected = { budget_id: "shared", max_budget: 42.57, budget_duration: "30d", tpm_limit: 500 };
    await waitFor(() => expect(requests.find((request) => request.path === "/budget/new")?.body).toEqual(expected));
  });

  it("requires at least one profile cap and a budget ID", async () => {
    const { requests } = setup();
    const user = userEvent.setup();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Budget type" }), "Per-person profile");
    await user.click(screen.getByRole("button", { name: "Create budget" }));
    expect(await screen.findByText("Set a monthly or annual cap. Zero is a valid cap.")).toBeVisible();
    expect(screen.getByLabelText("Budget ID")).toHaveAttribute("aria-invalid", "true");
    expect(requests.some((request) => request.path === "/budget/new")).toBe(false);
  });
});
