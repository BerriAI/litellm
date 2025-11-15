import * as networking from "../networking";
import { fireEvent, render, waitFor, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import BudgetPanel from "./budget_panel";

vi.mock("../networking", () => ({
  getBudgetList: vi.fn(),
  budgetDeleteCall: vi.fn(),
}));

describe("Budget Panel", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render the budget panel and load budgets", async () => {
    vi.mocked(networking.getBudgetList).mockResolvedValue([
      {
        budget_id: "budget-1",
        max_budget: "100",
        rpm_limit: 10,
        tpm_limit: 1000,
        updated_at: "2024-01-01T00:00:00Z",
      },
    ]);

    const { getByText } = render(<BudgetPanel accessToken="token-123" />);

    await waitFor(() => {
      expect(getByText("Create a budget to assign to customers.")).toBeInTheDocument();
      expect(getByText("budget-1")).toBeInTheDocument();
    });
  });

  it("should open delete modal when clicking delete icon", async () => {
    vi.mocked(networking.getBudgetList).mockResolvedValue([
      {
        budget_id: "budget-to-delete",
        max_budget: "200",
        rpm_limit: 20,
        tpm_limit: 2000,
        updated_at: "2024-01-02T00:00:00Z",
      },
    ]);

    const { getByText, container } = render(<BudgetPanel accessToken="token-123" />);

    await waitFor(() => {
      expect(getByText("budget-to-delete")).toBeInTheDocument();
    });

    // Find the first table row in tbody and click the second icon (trash/delete)
    const bodyRows = container.querySelectorAll("tbody tr");
    expect(bodyRows.length).toBeGreaterThan(0);
    const firstRow = bodyRows[0];
    const rowClickableIcons = firstRow.querySelectorAll(".cursor-pointer");
    expect(rowClickableIcons.length).toBeGreaterThan(1);

    fireEvent.click(rowClickableIcons[1]);

    await waitFor(() => {
      expect(screen.getByText("Delete Budget")).toBeInTheDocument();
    });
  });
});
