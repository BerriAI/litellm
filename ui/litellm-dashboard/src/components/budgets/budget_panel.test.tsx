import { fireEvent, render, waitFor, screen } from "@testing-library/react";
import { act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import BudgetPanel from "./budget_panel";

const mockBudgets = [
  {
    budget_id: "budget-1",
    max_budget: 100,
    rpm_limit: 10,
    tpm_limit: 1000,
    updated_at: "2024-01-01T00:00:00Z",
  },
];

vi.mock("@/app/(dashboard)/hooks/budgets/useBudgets", () => ({
  useBudgets: vi.fn().mockReturnValue({ data: [], isLoading: false }),
  useDeleteBudget: vi.fn().mockReturnValue({ mutateAsync: vi.fn(), isPending: false }),
  useCreateBudget: vi.fn().mockReturnValue({ mutateAsync: vi.fn() }),
  useUpdateBudget: vi.fn().mockReturnValue({ mutateAsync: vi.fn() }),
}));

import { useBudgets, useDeleteBudget, useCreateBudget, useUpdateBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });

function renderWithProviders(ui: React.ReactElement) {
  const qc = createQueryClient();
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

describe("Budget Panel", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render the budget panel and load budgets", async () => {
    vi.mocked(useBudgets).mockReturnValue({
      data: mockBudgets,
      isLoading: false,
    } as any);

    renderWithProviders(<BudgetPanel accessToken="token-123" />);

    await waitFor(() => {
      expect(screen.getByText("Create a budget to assign to customers.")).toBeInTheDocument();
      expect(screen.getByText("budget-1")).toBeInTheDocument();
    });
  });

  it("should open delete modal when clicking delete icon", async () => {
    vi.mocked(useBudgets).mockReturnValue({
      data: [
        {
          budget_id: "budget-to-delete",
          max_budget: 200,
          rpm_limit: 20,
          tpm_limit: 2000,
          updated_at: "2024-01-02T00:00:00Z",
        },
      ],
      isLoading: false,
    } as any);

    renderWithProviders(<BudgetPanel accessToken="token-123" />);

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
    const deleteMutateAsync = vi.fn().mockResolvedValue(undefined);
    vi.mocked(useBudgets).mockReturnValue({
      data: [
        {
          budget_id: "budget-to-delete",
          max_budget: 200,
          rpm_limit: 20,
          tpm_limit: 2000,
          updated_at: "2024-01-02T00:00:00Z",
        },
      ],
      isLoading: false,
    } as any);
    vi.mocked(useDeleteBudget).mockReturnValue({
      mutateAsync: deleteMutateAsync,
      isPending: false,
    } as any);

    renderWithProviders(<BudgetPanel accessToken="token-123" />);

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
      expect(deleteMutateAsync).toHaveBeenCalledWith("budget-to-delete");
    });
  });

  it("should render empty state without crashing", async () => {
    vi.mocked(useBudgets).mockReturnValue({
      data: [],
      isLoading: false,
    } as any);

    renderWithProviders(<BudgetPanel accessToken="token-123" />);

    await waitFor(() => {
      expect(screen.getByText("Create a budget to assign to customers.")).toBeInTheDocument();
    });
  });

  it("should handle delete error", async () => {
    const deleteMutateAsync = vi.fn().mockRejectedValue(new Error("Delete failed"));
    vi.mocked(useBudgets).mockReturnValue({
      data: [
        {
          budget_id: "budget-to-delete",
          max_budget: 200,
          rpm_limit: 20,
          tpm_limit: 2000,
          updated_at: "2024-01-02T00:00:00Z",
        },
      ],
      isLoading: false,
    } as any);
    vi.mocked(useDeleteBudget).mockReturnValue({
      mutateAsync: deleteMutateAsync,
      isPending: false,
    } as any);

    renderWithProviders(<BudgetPanel accessToken="token-123" />);

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
      expect(deleteMutateAsync).toHaveBeenCalledWith("budget-to-delete");
    });
  });

  it("should open edit modal when clicking edit icon", async () => {
    vi.mocked(useBudgets).mockReturnValue({
      data: [
        {
          budget_id: "budget-to-edit",
          max_budget: 300,
          rpm_limit: 30,
          tpm_limit: 3000,
          updated_at: "2024-01-03T00:00:00Z",
        },
      ],
      isLoading: false,
    } as any);

    renderWithProviders(<BudgetPanel accessToken="token-123" />);

    await waitFor(() => {
      expect(screen.getByText("budget-to-edit")).toBeInTheDocument();
    });

    const editButton = screen.getByTestId("edit-budget-button");

    act(() => {
      fireEvent.click(editButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Edit Budget")).toBeInTheDocument();
    });
  });
});
