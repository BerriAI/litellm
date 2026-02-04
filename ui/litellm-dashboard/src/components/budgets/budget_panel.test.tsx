import * as networking from "../networking";
import { fireEvent, render, waitFor, screen } from "@testing-library/react";
import { act } from "@testing-library/react";
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

    render(<BudgetPanel accessToken="token-123" />);

    await waitFor(() => {
      expect(screen.getByText("Create a budget to assign to customers.")).toBeInTheDocument();
      expect(screen.getByText("budget-1")).toBeInTheDocument();
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

    render(<BudgetPanel accessToken="token-123" />);

    await waitFor(() => {
      expect(screen.getByText("budget-to-delete")).toBeInTheDocument();
    });

    const deleteButton = screen.getByTestId("delete-budget-button");

    act(() => {
      fireEvent.click(deleteButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Delete Budget?")).toBeInTheDocument();
    });
  });

  it("should successfully delete a budget", async () => {
    vi.mocked(networking.getBudgetList).mockResolvedValue([
      {
        budget_id: "budget-to-delete",
        max_budget: "200",
        rpm_limit: 20,
        tpm_limit: 2000,
        updated_at: "2024-01-02T00:00:00Z",
      },
    ]);
    vi.mocked(networking.budgetDeleteCall).mockResolvedValue(undefined);

    render(<BudgetPanel accessToken="token-123" />);

    await waitFor(() => {
      expect(screen.getByText("budget-to-delete")).toBeInTheDocument();
    });

    // Open delete modal
    const deleteButton = screen.getByTestId("delete-budget-button");
    act(() => {
      fireEvent.click(deleteButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Delete Budget?")).toBeInTheDocument();
    });

    // Confirm delete
    const confirmButton = screen.getByRole("button", { name: /delete/i });
    act(() => {
      fireEvent.click(confirmButton);
    });

    await waitFor(() => {
      expect(networking.budgetDeleteCall).toHaveBeenCalledWith("token-123", "budget-to-delete");
      expect(networking.getBudgetList).toHaveBeenCalledTimes(2); // Initial load + refresh after delete
    });
  });

  it("should handle delete error", async () => {
    vi.mocked(networking.getBudgetList).mockResolvedValue([
      {
        budget_id: "budget-to-delete",
        max_budget: "200",
        rpm_limit: 20,
        tpm_limit: 2000,
        updated_at: "2024-01-02T00:00:00Z",
      },
    ]);
    vi.mocked(networking.budgetDeleteCall).mockRejectedValue(new Error("Delete failed"));

    render(<BudgetPanel accessToken="token-123" />);

    await waitFor(() => {
      expect(screen.getByText("budget-to-delete")).toBeInTheDocument();
    });

    // Open delete modal
    const deleteButton = screen.getByTestId("delete-budget-button");
    act(() => {
      fireEvent.click(deleteButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Delete Budget?")).toBeInTheDocument();
    });

    // Confirm delete
    const confirmButton = screen.getByRole("button", { name: /delete/i });
    act(() => {
      fireEvent.click(confirmButton);
    });

    await waitFor(() => {
      expect(networking.budgetDeleteCall).toHaveBeenCalledWith("token-123", "budget-to-delete");
    });

    // Modal should still be open (error handling)
    expect(screen.getByText("Delete Budget?")).toBeInTheDocument();
  });
});
