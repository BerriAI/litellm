import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import BudgetTable from "./BudgetTable";
import { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";

const makeBudget = (overrides: Partial<budgetItem> = {}): budgetItem => ({
  budget_id: "budget-1",
  max_budget: 100,
  tpm_limit: 1000,
  rpm_limit: 10,
  updated_at: "2024-01-01T00:00:00Z",
  ...overrides,
});

const defaultProps = {
  budgets: [makeBudget()],
  isLoading: false,
  canModify: true,
  onEditClick: vi.fn(),
  onDeleteClick: vi.fn(),
};

describe("BudgetTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should display budget information", () => {
    renderWithProviders(<BudgetTable {...defaultProps} />);
    expect(screen.getByText("budget-1")).toBeInTheDocument();
    expect(screen.getByText("$100.00")).toBeInTheDocument();
    expect(screen.getByText("1000")).toBeInTheDocument();
    expect(screen.getByText("10")).toBeInTheDocument();
  });

  it("should show n/a for missing rate limits and Unlimited for a missing max budget", () => {
    renderWithProviders(
      <BudgetTable {...defaultProps} budgets={[makeBudget({ max_budget: null, tpm_limit: null, rpm_limit: null })]} />,
    );
    expect(screen.getAllByText("n/a")).toHaveLength(2);
    expect(screen.getByText("Unlimited")).toBeInTheDocument();
  });

  it("should sort budgets by updated_at descending", () => {
    const budgets = [
      makeBudget({ budget_id: "budget-old", updated_at: "2024-01-01T00:00:00Z" }),
      makeBudget({ budget_id: "budget-new", updated_at: "2024-06-01T00:00:00Z" }),
    ];
    renderWithProviders(<BudgetTable {...defaultProps} budgets={budgets} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("budget-new")).toBeInTheDocument();
    expect(within(rows[1]).getByText("budget-old")).toBeInTheDocument();
  });

  it("should call onEditClick from the actions menu", async () => {
    const user = userEvent.setup();
    renderWithProviders(<BudgetTable {...defaultProps} />);
    await user.click(screen.getByTestId("budget-actions-budget-1"));
    await user.click(await screen.findByTestId("budget-action-edit"));
    expect(defaultProps.onEditClick).toHaveBeenCalledWith(defaultProps.budgets[0]);
  });

  it("should call onDeleteClick from the actions menu", async () => {
    const user = userEvent.setup();
    renderWithProviders(<BudgetTable {...defaultProps} />);
    await user.click(screen.getByTestId("budget-actions-budget-1"));
    await user.click(await screen.findByTestId("budget-action-delete"));
    expect(defaultProps.onDeleteClick).toHaveBeenCalledWith(defaultProps.budgets[0]);
  });

  it("should not render the actions menu when the user cannot modify budgets", () => {
    renderWithProviders(<BudgetTable {...defaultProps} canModify={false} />);
    expect(screen.queryByTestId("budget-actions-budget-1")).not.toBeInTheDocument();
  });

  it("should show skeleton rows when loading", () => {
    renderWithProviders(<BudgetTable {...defaultProps} budgets={[]} isLoading />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
  });

  it("should show the empty state when there are no budgets", () => {
    renderWithProviders(<BudgetTable {...defaultProps} budgets={[]} />);
    expect(screen.getByText("No budgets yet")).toBeInTheDocument();
  });
});
