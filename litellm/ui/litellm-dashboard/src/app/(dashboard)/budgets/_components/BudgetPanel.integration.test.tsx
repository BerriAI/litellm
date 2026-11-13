import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { budgetTestNetwork, jsonResponse } from "@/../tests/budgetTestNetwork";
import BudgetPanel from "./BudgetPanel";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
vi.unmock("@/lib/toast");
afterEach(() => vi.unstubAllGlobals());

describe("BudgetPanel profile wiring", () => {
  it("lists independent caps and assignments and opens the populated profile editor", async () => {
    const profile = {
      budget_id: "starter",
      is_profile: true,
      revision: "loaded",
      assignment_count: 7,
      budget_limits: [
        { budget_duration: "1mo", max_budget: 0 },
        { budget_duration: "12mo", max_budget: 75 },
      ],
      cap_policy: "pause_until_reset",
      max_budget: null,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    };
    const envelope = { data: [profile], meta: { total_count: 1, page: 1, page_size: 50, total_pages: 1 } };
    const { client } = budgetTestNetwork(() => jsonResponse(envelope));
    render(
      <QueryClientProvider client={client}>
        <BudgetPanel accessToken="budget-test-token" />
      </QueryClientProvider>,
    );
    expect(await screen.findByText("$0 / Monthly")).toBeVisible();
    expect(screen.getByText("$75 / Annual")).toBeVisible();
    expect(screen.getByText("7")).toBeVisible();
    expect(await screen.findByRole("button", { name: "Assign budgets" })).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "Open budget actions" }));
    await userEvent.click(await screen.findByRole("menuitem", { name: "Edit budget" }));
    expect(await screen.findByLabelText("Monthly cap (USD)")).toHaveValue(0);
    expect(screen.getByLabelText("Annual cap (USD)")).toHaveValue(75);
  });
});
