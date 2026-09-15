import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import BudgetTable from "./BudgetTable";
import type { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import type { ResourceListResult } from "@/app/(dashboard)/hooks/common/useResourceList";
import { ApiError } from "@/lib/http/client";

const { copyToClipboardMock } = vi.hoisted(() => ({ copyToClipboardMock: vi.fn() }));

vi.mock("@/utils/dataUtils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/dataUtils")>()),
  copyToClipboard: copyToClipboardMock,
}));

const makeBudget = (overrides: Partial<budgetItem> = {}): budgetItem => ({
  budget_id: "budget-1",
  max_budget: 100,
  soft_budget: null,
  tpm_limit: 1000,
  rpm_limit: 10,
  budget_duration: "30d",
  budget_reset_at: null,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  ...overrides,
});

const makeList = (overrides: Partial<ResourceListResult<budgetItem>> = {}): ResourceListResult<budgetItem> => ({
  rows: [makeBudget()],
  rowCount: 1,
  isLoading: false,
  isFetching: false,
  error: null,
  refetch: vi.fn(),
  sorting: [{ id: "created_at", desc: true }],
  onSortingChange: vi.fn(),
  pagination: { pageIndex: 0, pageSize: 50 },
  onPaginationChange: vi.fn(),
  columnFilters: [],
  onColumnFiltersChange: vi.fn(),
  searchValue: "",
  onSearchChange: vi.fn(),
  ...overrides,
});

const FORBIDDEN_PROBLEM = {
  type: "about:blank",
  title: "Forbidden",
  status: 403,
  detail: "Only proxy admins can view budgets",
};

const showColumn = async (user: ReturnType<typeof userEvent.setup>, columnId: string) => {
  await user.click(screen.getByTestId("view-options-trigger"));
  await user.click(await screen.findByTestId(`view-option-${columnId}`));
};

const defaultProps = {
  canModify: true,
  onEditClick: vi.fn(),
  onDeleteClick: vi.fn(),
};

describe("BudgetTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
  });

  it("should display budget information", () => {
    renderWithProviders(<BudgetTable {...defaultProps} list={makeList()} />);
    expect(screen.getByText("budget-1")).toBeInTheDocument();
    expect(screen.getByText("$100.00")).toBeInTheDocument();
    expect(screen.getByText("1000")).toBeInTheDocument();
    expect(screen.getByText("10")).toBeInTheDocument();
  });

  it("should open on the four columns the page has always shown, with reset and created off", () => {
    renderWithProviders(<BudgetTable {...defaultProps} list={makeList()} />);
    const headers = screen.getAllByRole("columnheader").map((header) => header.textContent);
    expect(headers).toEqual(expect.arrayContaining(["Budget ID", "Max Budget", "TPM", "RPM"]));
    expect(headers).not.toContain("Reset");
    expect(headers).not.toContain("Created");
  });

  it("should render the reset column with the friendly duration label once it is turned on", async () => {
    const user = userEvent.setup();
    renderWithProviders(<BudgetTable {...defaultProps} list={makeList()} />);
    await showColumn(user, "budget_duration");
    expect(screen.getByText("monthly")).toBeInTheDocument();
  });

  it("should render 'Not set' when a budget has no reset duration", async () => {
    const user = userEvent.setup();
    const list = makeList({ rows: [makeBudget({ budget_duration: null })] });
    renderWithProviders(<BudgetTable {...defaultProps} list={list} />);
    await showColumn(user, "budget_duration");
    expect(screen.getByText("Not set")).toBeInTheDocument();
  });

  it("should render the budget id in full, with no truncation", () => {
    const budgetId = "ecc1869c-6231-4380-a56d-1a0be457477d";
    const list = makeList({ rows: [makeBudget({ budget_id: budgetId })] });
    renderWithProviders(<BudgetTable {...defaultProps} list={list} />);
    const idCell = screen.getByText(budgetId);
    expect(idCell).not.toHaveClass("truncate");
    expect(idCell.className).not.toMatch(/max-w-\[\d+(ch|rem|px)\]/);
  });

  it("should keep the budget id on a single line", () => {
    const budgetId = "ecc1869c-6231-4380-a56d-1a0be457477d";
    const list = makeList({ rows: [makeBudget({ budget_id: budgetId })] });
    renderWithProviders(<BudgetTable {...defaultProps} list={list} />);
    expect(screen.getByText(budgetId)).toHaveClass("whitespace-nowrap");
  });

  it("should copy the budget id from the cell's copy button", async () => {
    const user = userEvent.setup();
    const budgetId = "ecc1869c-6231-4380-a56d-1a0be457477d";
    const list = makeList({ rows: [makeBudget({ budget_id: budgetId })] });
    renderWithProviders(<BudgetTable {...defaultProps} list={list} />);
    await user.click(screen.getByRole("button", { name: "Copy ID" }));
    expect(copyToClipboardMock).toHaveBeenCalledWith(budgetId);
  });

  it("should offer sorting on every backend-sortable column", async () => {
    const user = userEvent.setup();
    renderWithProviders(<BudgetTable {...defaultProps} list={makeList()} />);
    await showColumn(user, "created_at");
    for (const field of ["budget_id", "max_budget", "tpm_limit", "rpm_limit", "created_at"]) {
      expect(screen.getByTestId(`sort-header-${field}`)).toBeInTheDocument();
    }
  });

  it("should not make the reset column sortable", async () => {
    const user = userEvent.setup();
    renderWithProviders(<BudgetTable {...defaultProps} list={makeList()} />);
    await showColumn(user, "budget_duration");
    const headers = screen.getAllByRole("columnheader").map((header) => header.textContent);
    expect(headers).toContain("Reset");
    expect(screen.queryByTestId("sort-header-budget_duration")).not.toBeInTheDocument();
  });

  it("should ask the list for a new sort when a sortable header is clicked", async () => {
    const user = userEvent.setup();
    const onSortingChange = vi.fn();
    renderWithProviders(<BudgetTable {...defaultProps} list={makeList({ onSortingChange })} />);
    await user.click(screen.getByTestId("sort-header-max_budget"));
    expect(onSortingChange).toHaveBeenCalled();
  });

  it("should show n/a for missing rate limits and Unlimited for a missing max budget", () => {
    const list = makeList({ rows: [makeBudget({ max_budget: null, tpm_limit: null, rpm_limit: null })] });
    renderWithProviders(<BudgetTable {...defaultProps} list={list} />);
    expect(screen.getAllByText("n/a")).toHaveLength(2);
    expect(screen.getByText("Unlimited")).toBeInTheDocument();
  });

  it("should call onEditClick from the actions menu", async () => {
    const user = userEvent.setup();
    const list = makeList();
    renderWithProviders(<BudgetTable {...defaultProps} list={list} />);
    await user.click(screen.getByTestId("budget-actions-budget-1"));
    await user.click(await screen.findByTestId("budget-action-edit"));
    expect(defaultProps.onEditClick).toHaveBeenCalledWith(list.rows[0]);
  });

  it("should call onDeleteClick from the actions menu", async () => {
    const user = userEvent.setup();
    const list = makeList();
    renderWithProviders(<BudgetTable {...defaultProps} list={list} />);
    await user.click(screen.getByTestId("budget-actions-budget-1"));
    await user.click(await screen.findByTestId("budget-action-delete"));
    expect(defaultProps.onDeleteClick).toHaveBeenCalledWith(list.rows[0]);
  });

  it("should not render the actions menu when the user cannot modify budgets", () => {
    renderWithProviders(<BudgetTable {...defaultProps} canModify={false} list={makeList()} />);
    expect(screen.queryByTestId("budget-actions-budget-1")).not.toBeInTheDocument();
  });

  it("should show skeleton rows when loading", () => {
    renderWithProviders(<BudgetTable {...defaultProps} list={makeList({ rows: [], isLoading: true })} />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
  });

  it("should show the empty state when there are no budgets", () => {
    renderWithProviders(<BudgetTable {...defaultProps} list={makeList({ rows: [], rowCount: 0 })} />);
    expect(screen.getByText("No budgets yet")).toBeInTheDocument();
  });

  it("should tell the user their search matched nothing rather than that no budgets exist", () => {
    const list = makeList({ rows: [], rowCount: 0, searchValue: "nope" });
    renderWithProviders(<BudgetTable {...defaultProps} list={list} />);
    expect(screen.getByText("No matching budgets")).toBeInTheDocument();
  });

  it("should render an access-denied state for a 403 instead of an empty table", () => {
    const error = new ApiError("Only proxy admins can view budgets", 403, FORBIDDEN_PROBLEM);
    const list = makeList({ rows: [], rowCount: 0, error });
    const { container } = renderWithProviders(<BudgetTable {...defaultProps} list={list} />);
    expect(screen.getByText("You do not have access to budgets")).toBeInTheDocument();
    expect(screen.queryByText("No budgets yet")).not.toBeInTheDocument();
    expect(container.querySelector(".lucide-shield-alert")).not.toBeNull();
  });

  it("should surface the problem detail for a non-403 failure", () => {
    const error = new ApiError("budget store unavailable", 500, null);
    const list = makeList({ rows: [], rowCount: 0, error });
    renderWithProviders(<BudgetTable {...defaultProps} list={list} />);
    expect(screen.getByText("Could not load budgets")).toBeInTheDocument();
    expect(screen.getByText("budget store unavailable")).toBeInTheDocument();
  });
});
