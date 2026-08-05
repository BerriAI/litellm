import type { ColumnFiltersState, PaginationState } from "@tanstack/react-table";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AuditLogsTable } from "./AuditLogsTable";
import type { AuditLogEntry } from "./AuditLogsTableColumns";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

vi.mock("../networking", () => ({
  serverRootPath: "",
}));

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
    object_alias: "prod-team",
    changed_by_user_email: null,
    changed_by_key_alias: null,
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
    changed_by_user_email: "admin@example.com",
    changed_by_key_alias: "admin-key",
  },
];

const makeRow = (overrides: Partial<AuditLogEntry> & Pick<AuditLogEntry, "id" | "table_name" | "object_id">) => ({
  updated_at: "2026-07-20T10:00:00Z",
  changed_by: "user-42",
  changed_by_api_key: "sk-hash-xyz",
  action: "updated",
  before_value: {},
  updated_values: {},
  ...overrides,
});

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

  it("renders the alias column with the object alias and an em-dash fallback when absent", () => {
    renderTable();

    expect(screen.getByText("prod-team")).toBeInTheDocument();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("shows the changed-by email and key alias while keeping the raw id and hash visible", () => {
    renderTable();

    expect(screen.getByText("admin@example.com")).toBeInTheDocument();
    expect(screen.getByText("user-42")).toBeInTheDocument();
    expect(screen.getByText("admin-key")).toBeInTheDocument();
    expect(screen.getByText("sk-hash-def")).toBeInTheDocument();
  });

  it("links object ids and changed-by users to their entity detail pages", () => {
    renderTable({
      data: [
        ...ROWS,
        makeRow({ id: "log-3", table_name: "LiteLLM_VerificationToken", object_id: "keyhash-1" }),
        makeRow({ id: "log-4", table_name: "LiteLLM_OrganizationTable", object_id: "org-1" }),
        makeRow({ id: "log-5", table_name: "LiteLLM_ProxyModelTable", object_id: "model-1" }),
        makeRow({ id: "log-6", table_name: "SomeUnknownTable", object_id: "mystery-1" }),
      ],
      rowCount: 6,
    });

    expect(screen.getByRole("link", { name: "team-obj-123" })).toHaveAttribute(
      "href",
      expect.stringContaining("/teams?team=team-obj-123"),
    );
    expect(screen.getByRole("link", { name: "user-obj-456" })).toHaveAttribute(
      "href",
      expect.stringContaining("/users?user=user-obj-456"),
    );
    expect(screen.getByRole("link", { name: "keyhash-1" })).toHaveAttribute(
      "href",
      expect.stringContaining("/api-keys?key=keyhash-1"),
    );
    expect(screen.getByRole("link", { name: "org-1" })).toHaveAttribute(
      "href",
      expect.stringContaining("/organizations?org=org-1"),
    );
    expect(screen.getByRole("link", { name: "model-1" })).toHaveAttribute(
      "href",
      expect.stringContaining("/models-and-endpoints?model=model-1"),
    );
    expect(screen.getAllByRole("link", { name: "admin@example.com" })[0]).toHaveAttribute(
      "href",
      expect.stringContaining("/users?user=user-42"),
    );
    expect(screen.queryByRole("link", { name: "mystery-1" })).toBeNull();
    expect(screen.getByText("mystery-1")).toBeInTheDocument();
  });

  it("opens the detail drawer from a row click with the full row", async () => {
    const user = userEvent.setup();
    const props = renderTable();

    await user.click(screen.getByText("Teams"));

    expect(props.onViewLog).toHaveBeenCalledTimes(1);
    expect(props.onViewLog).toHaveBeenCalledWith(ROWS[0]);
  });

  it("does not open the drawer when the object id entity link is clicked", async () => {
    const user = userEvent.setup();
    const props = renderTable();

    await user.click(screen.getByRole("link", { name: "team-obj-123" }));

    expect(props.onViewLog).not.toHaveBeenCalled();
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

  it("commits the team filter as an object_team filter accepting id or alias", async () => {
    const user = userEvent.setup();
    const onColumnFiltersChange = vi.fn();
    renderTable({ onColumnFiltersChange });

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await user.type(await screen.findByPlaceholderText("Team ID or alias…"), "ml-platform");
    await user.click(screen.getByTestId("filter-drawer-apply"));

    const arg = onColumnFiltersChange.mock.calls[0][0];
    const committed = typeof arg === "function" ? arg([]) : arg;
    expect(committed).toEqual([{ id: "object_team", value: "ml-platform" }]);
  });

  it("commits the date range as start_date and end_date filters", async () => {
    const user = userEvent.setup();
    const onColumnFiltersChange = vi.fn();
    renderTable({ onColumnFiltersChange });

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    fireEvent.change(await screen.findByTestId("audit-filter-start-date"), { target: { value: "2026-07-01T10:00" } });
    fireEvent.change(screen.getByTestId("audit-filter-end-date"), { target: { value: "2026-07-02T18:30" } });
    await user.click(screen.getByTestId("filter-drawer-apply"));

    const arg = onColumnFiltersChange.mock.calls[0][0];
    const committed = typeof arg === "function" ? arg([]) : arg;
    expect(committed).toEqual(
      expect.arrayContaining([
        { id: "start_date", value: "2026-07-01T10:00" },
        { id: "end_date", value: "2026-07-02T18:30" },
      ]),
    );
  });
});
