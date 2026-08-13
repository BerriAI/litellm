import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, it, expect, beforeEach } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import { DeletedKeysTable } from "./DeletedKeysTable";
import { DeletedKeyResponse } from "@/app/(dashboard)/hooks/keys/useKeys";

const makeDeletedKey = (overrides: Partial<DeletedKeyResponse> = {}): DeletedKeyResponse =>
  ({
    token: "sk-1234567890abcdef",
    token_id: "key-1",
    key_name: "test-key",
    key_alias: "Test Key Alias",
    spend: 5.5,
    max_budget: 100,
    models: ["gpt-3.5-turbo"],
    user_id: "user-1",
    team_id: "team-1",
    organization_id: "org-1",
    created_at: "2024-11-01T10:00:00Z",
    updated_at: "2024-11-15T10:00:00Z",
    created_by: "creator-1",
    team_alias: "Test Team",
    user_email: "user@example.com",
    deleted_at: "2024-11-15T10:00:00Z",
    deleted_by: "user-1",
    ...overrides,
  }) as DeletedKeyResponse;

const defaultProps = {
  keys: [makeDeletedKey()],
  totalCount: 1,
  isLoading: false,
  pagination: { pageIndex: 0, pageSize: 50 },
  onPaginationChange: vi.fn(),
};

beforeEach(() => {
  vi.clearAllMocks();
});

it("should display key information", () => {
  renderWithProviders(<DeletedKeysTable {...defaultProps} />);

  expect(screen.getByText("Test Key Alias")).toBeInTheDocument();
  expect(screen.getByText("sk-1234567890abcdef")).toBeInTheDocument();
  expect(screen.getByText("user@example.com")).toBeInTheDocument();
});

it("should show the total count in the pagination footer", () => {
  renderWithProviders(<DeletedKeysTable {...defaultProps} totalCount={120} />);

  expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-50 of 120");
  expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 3");
});

it("should propagate pagination changes when the next page button is clicked", async () => {
  const user = userEvent.setup();
  renderWithProviders(<DeletedKeysTable {...defaultProps} totalCount={120} />);

  await user.click(screen.getByTestId("pagination-next"));

  expect(defaultProps.onPaginationChange).toHaveBeenCalled();
});

it("should sort the current page by deleted_at descending by default", () => {
  const keys = [
    makeDeletedKey({ token: "sk-older", key_alias: "older-key", deleted_at: "2024-01-01T10:00:00Z" }),
    makeDeletedKey({ token: "sk-newer", key_alias: "newer-key", deleted_at: "2024-06-01T10:00:00Z" }),
  ];
  renderWithProviders(<DeletedKeysTable {...defaultProps} keys={keys} totalCount={2} />);

  const rows = screen.getAllByRole("row").slice(1);
  expect(within(rows[0]).getByText("newer-key")).toBeInTheDocument();
  expect(within(rows[1]).getByText("older-key")).toBeInTheDocument();
});

it("should show skeleton rows when loading", () => {
  renderWithProviders(<DeletedKeysTable {...defaultProps} keys={[]} isLoading />);

  expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
});

it("should show the empty state when there are no deleted keys", () => {
  renderWithProviders(<DeletedKeysTable {...defaultProps} keys={[]} totalCount={0} />);

  expect(screen.getByText("No deleted keys found")).toBeInTheDocument();
});
