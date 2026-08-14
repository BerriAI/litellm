import type { ColumnFiltersState, PaginationState } from "@tanstack/react-table";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AuditLogsTable } from "./AuditLogsTable";
import type { AuditLogEntry } from "./AuditLogsTableColumns";

const ROWS: AuditLogEntry[] = [
  {
    id: "log-1",
    updated_at: "2026-07-20T12:00:00Z",
    changed_by: "default_user_id",
    changed_by_api_key: "sk-hash-abc",
    action: "created",
    table_name: "LiteLLM_TeamTable",
    object_id: "team-obj-123",
    before_value: {},
    updated_values: { foo: "bar" },
  },
  {
    id: "log-2",
    updated_at: "2026-07-20T11:00:00Z",
    changed_by: "user-42",
    changed_by_api_key: "sk-hash-def",
    action: "deleted",
    table_name: "LiteLLM_UserTable",
    object_id: "user-obj-456",
    before_value: { a: 1 },
    updated_values: {},
  },
];

const FIRST_PAGE: PaginationState = { pageIndex: 0, pageSize: 50 };

function renderTable(overrides: Partial<React.ComponentProps<typeof AuditLogsTable>> = {}) {
  const props: React.ComponentProps<typeof AuditLogsTable> = {
    data: ROWS,
    rowCount: ROWS.length,
    isLoading: false,
    isRefreshing: false,
    pagination: FIRST_PAGE,
    onPaginationChange: vi.fn(),
    columnFilters: [],
    onColumnFiltersChange: vi.fn(),
    onRefresh: vi.fn(),
    onViewLog: vi.fn(),
    ...overrides,
  };
  render(<AuditLogsTable {...props} />);
  return props;
}

describe("AuditLogsTable", () => {
  it("renders each audit column with the migrated shared cells", () => {
    renderTable();

    // Action -> StatusBadge with a capitalized label
    expect(screen.getByText("Created")).toBeInTheDocument();
    expect(screen.getByText("Deleted")).toBeInTheDocument();
    // Table name -> display mapping
    expect(screen.getByText("Teams")).toBeInTheDocument();
    expect(screen.getByText("Users")).toBeInTheDocument();
    // Changed By -> DefaultProxyAdminTag (default_user_id becomes a labeled tag; other ids stay raw)
    expect(screen.getByText("Default Proxy Admin")).toBeInTheDocument();
    expect(screen.getByText("user-42")).toBeInTheDocument();
    // Object ID + API key hash
    expect(screen.getByText("team-obj-123")).toBeInTheDocument();
    expect(screen.getByText("sk-hash-abc")).toBeInTheDocument();
  });

  it("opens the detail drawer from the Object ID identity cell with the full row", async () => {
    const user = userEvent.setup();
    const props = renderTable();

    await user.click(screen.getByText("team-obj-123"));

    expect(props.onViewLog).toHaveBeenCalledTimes(1);
    expect(props.onViewLog).toHaveBeenCalledWith(ROWS[0]);
  });

  it("drives the shared footer from the server rowCount and reports page changes", async () => {
    const user = userEvent.setup();
    const onPaginationChange = vi.fn();
    renderTable({ rowCount: 120, onPaginationChange });

    // ceil(120 / 50) = 3 pages, proving rowCount (not data length) feeds the footer
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 3");

    await user.click(screen.getByTestId("pagination-next"));
    expect(onPaginationChange).toHaveBeenCalledTimes(1);
  });

  it("shows skeleton rows while loading and no data rows", () => {
    renderTable({ isLoading: true, data: [] });

    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("No audit logs yet")).toBeNull();
  });

  it("uses a distinct empty state for unfiltered vs filtered-empty results", () => {
    const { unmount } = render(
      <AuditLogsTable
        data={[]}
        rowCount={0}
        isLoading={false}
        isRefreshing={false}
        pagination={FIRST_PAGE}
        onPaginationChange={vi.fn()}
        columnFilters={[]}
        onColumnFiltersChange={vi.fn()}
        onRefresh={vi.fn()}
        onViewLog={vi.fn()}
      />,
    );
    expect(screen.getByText("No audit logs yet")).toBeInTheDocument();
    unmount();

    renderTable({ data: [], rowCount: 0, columnFilters: [{ id: "action", value: "created" }] });
    expect(screen.getByText("No matching audit logs")).toBeInTheDocument();
  });

  it("renders active filter chips with human-readable labels", () => {
    const filters: ColumnFiltersState = [{ id: "action", value: "created" }];
    renderTable({ columnFilters: filters });

    const chip = screen.getByTestId("filter-chip-action");
    expect(chip).toHaveTextContent("Action:");
    expect(chip).toHaveTextContent("Created");
  });

  it("commits a text filter through the filter drawer and reports it to the parent", async () => {
    const user = userEvent.setup();
    const onColumnFiltersChange = vi.fn();
    renderTable({ onColumnFiltersChange });

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await user.type(await screen.findByPlaceholderText("Enter object ID…"), "obj-9");
    await user.click(screen.getByTestId("filter-drawer-apply"));

    expect(onColumnFiltersChange).toHaveBeenCalledTimes(1);
    const arg = onColumnFiltersChange.mock.calls[0][0];
    const committed = typeof arg === "function" ? arg([]) : arg;
    expect(committed).toEqual([{ id: "object_id", value: "obj-9" }]);
  });
});
