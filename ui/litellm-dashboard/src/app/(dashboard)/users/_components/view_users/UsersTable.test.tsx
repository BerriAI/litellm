/* @vitest-environment jsdom */
import type { PaginationState, RowSelectionState, SortingState } from "@tanstack/react-table";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { UserInfo } from "@/components/networking";

import { UsersTable } from "./UsersTable";

const possibleUIRoles = {
  proxy_admin: { ui_label: "Admin" },
  internal_user: { ui_label: "Internal User" },
};

const makeUser = (overrides: Partial<UserInfo> = {}): UserInfo =>
  ({
    user_id: "user-1",
    user_email: "ada@example.com",
    user_alias: null,
    user_role: "proxy_admin",
    spend: 12.5,
    max_budget: null,
    models: [],
    key_count: 2,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-02-01T00:00:00Z",
    sso_user_id: null,
    budget_duration: null,
    ...overrides,
  }) as UserInfo;

interface HarnessOverrides {
  data?: UserInfo[];
  rowCount?: number;
  isLoading?: boolean;
  selectionEnabled?: boolean;
  onUserClick?: (userId: string, openInEditMode?: boolean) => void;
  onDeleteUser?: (user: UserInfo) => void;
  onResetPassword?: (userId: string) => void;
  onSortingChange?: ReturnType<typeof vi.fn>;
}

/**
 * Renders the table with real selection/sorting state so assertions exercise the
 * controlled wiring rather than a stubbed callback.
 */
function Harness({
  data = [makeUser()],
  rowCount = 1,
  isLoading = false,
  selectionEnabled = false,
  onUserClick = vi.fn(),
  onDeleteUser = vi.fn(),
  onResetPassword = vi.fn(),
  onSortingChange,
}: HarnessOverrides) {
  const [sorting, setSorting] = useState<SortingState>([{ id: "created_at", desc: true }]);
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 25 });
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});

  return (
    <>
      <span data-testid="selected-ids">
        {Object.keys(rowSelection)
          .filter((key) => rowSelection[key])
          .sort()
          .join(",")}
      </span>
      <UsersTable
        data={data}
        rowCount={rowCount}
        isLoading={isLoading}
        possibleUIRoles={possibleUIRoles}
        teams={[]}
        sorting={sorting}
        onSortingChange={(updater) => {
          setSorting(updater);
          onSortingChange?.(updater);
        }}
        pagination={pagination}
        onPaginationChange={setPagination}
        columnFilters={[]}
        onColumnFiltersChange={vi.fn()}
        searchValue=""
        onSearchChange={vi.fn()}
        selectionEnabled={selectionEnabled}
        rowSelection={rowSelection}
        onRowSelectionChange={setRowSelection}
        onUserClick={onUserClick}
        onDeleteUser={onDeleteUser}
        onResetPassword={onResetPassword}
      />
    </>
  );
}

const openRowMenu = async (user: ReturnType<typeof userEvent.setup>, userId: string) => {
  await user.click(screen.getByTestId(`user-actions-${userId}`));
};

describe("UsersTable", () => {
  it("renders every migrated column header", () => {
    render(<Harness />);

    const headerRow = screen.getAllByRole("row")[0];

    [
      "User ID",
      "Email",
      "Status",
      "Global Proxy Role",
      "User Alias",
      "Spend (USD)",
      "Budget (USD)",
      "SSO ID",
      "Virtual Keys",
      "Created At",
      "Updated At",
    ].forEach((header) => {
      expect(headerRow.textContent).toContain(header);
    });
  });

  // Sorting is server-side and the backend only accepts these five keys, so a sort
  // control on any other column would send an invalid sort_by. Assert the exact set:
  // a missing control and an extra one both have to fail.
  it("exposes a sort control for exactly the five server-sortable columns", () => {
    render(<Harness />);

    const sortableIds = screen
      .getAllByTestId(/^sort-header-/)
      .map((node) => (node.getAttribute("data-testid") ?? "").replace("sort-header-", ""))
      .sort();

    expect(sortableIds).toEqual(["created_at", "spend", "user_email", "user_id", "user_role"]);
  });

  it("reports the clicked column to the server sorting handler", async () => {
    const user = userEvent.setup();
    const onSortingChange = vi.fn();
    render(<Harness onSortingChange={onSortingChange} />);

    await user.click(screen.getByTestId("sort-header-user_email"));

    expect(onSortingChange).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("sort-header-user_email").querySelector("[data-sort-indicator]")).toHaveAttribute(
      "data-sort-indicator",
      "asc",
    );
  });

  it("opens the detail view from the identity cell without edit mode", async () => {
    const user = userEvent.setup();
    const onUserClick = vi.fn();
    render(<Harness onUserClick={onUserClick} />);

    await user.click(screen.getByRole("button", { name: /user-1/ }));

    expect(onUserClick).toHaveBeenCalledWith("user-1", false);
  });

  it("opens the detail view in edit mode from the row menu", async () => {
    const user = userEvent.setup();
    const onUserClick = vi.fn();
    render(<Harness onUserClick={onUserClick} />);

    await openRowMenu(user, "user-1");
    await user.click(await screen.findByTestId("user-action-edit"));

    expect(onUserClick).toHaveBeenCalledWith("user-1", true);
  });

  it("delegates delete and reset-password from the row menu", async () => {
    const user = userEvent.setup();
    const onDeleteUser = vi.fn();
    const onResetPassword = vi.fn();
    render(<Harness onDeleteUser={onDeleteUser} onResetPassword={onResetPassword} />);

    await openRowMenu(user, "user-1");
    await user.click(await screen.findByTestId("user-action-reset-password"));
    expect(onResetPassword).toHaveBeenCalledWith("user-1");

    await openRowMenu(user, "user-1");
    await user.click(await screen.findByTestId("user-action-delete"));
    expect(onDeleteUser).toHaveBeenCalledWith(expect.objectContaining({ user_id: "user-1" }));
  });

  it("renders the SCIM status cell from metadata", () => {
    const { rerender } = render(<Harness data={[makeUser()]} />);
    expect(screen.getByTestId("user-status-user-1")).toHaveTextContent("Active");

    rerender(<Harness data={[makeUser({ metadata: { scim_active: false } } as Partial<UserInfo>)]} />);
    expect(screen.getByTestId("user-status-user-1")).toHaveTextContent("Inactive");

    rerender(<Harness data={[makeUser({ metadata: { scim_active: true } } as Partial<UserInfo>)]} />);
    expect(screen.getByTestId("user-status-user-1")).toHaveTextContent("Active");
  });

  describe("row selection", () => {
    const twoUsers = [
      makeUser({ user_id: "user-1", user_email: "ada@example.com" }),
      makeUser({ user_id: "user-2", user_email: "grace@example.com" }),
    ];

    it("hides the selection column until selection mode is on", () => {
      const { rerender } = render(<Harness data={twoUsers} rowCount={2} selectionEnabled={false} />);
      expect(screen.queryByTestId("datatable-select-all")).not.toBeInTheDocument();
      expect(screen.queryByTestId("datatable-select-row-user-1")).not.toBeInTheDocument();

      rerender(<Harness data={twoUsers} rowCount={2} selectionEnabled />);
      expect(screen.getByTestId("datatable-select-all")).toBeInTheDocument();
      expect(screen.getByTestId("datatable-select-row-user-1")).toBeInTheDocument();
    });

    it("keys the controlled selection by user id", async () => {
      const user = userEvent.setup();
      render(<Harness data={twoUsers} rowCount={2} selectionEnabled />);

      await user.click(screen.getByTestId("datatable-select-row-user-2"));
      expect(screen.getByTestId("selected-ids")).toHaveTextContent("user-2");

      await user.click(screen.getByTestId("datatable-select-row-user-1"));
      expect(screen.getByTestId("selected-ids")).toHaveTextContent("user-1,user-2");

      await user.click(screen.getByTestId("datatable-select-row-user-2"));
      expect(screen.getByTestId("selected-ids")).toHaveTextContent("user-1");
    });

    it("selects and clears the whole page from the header checkbox", async () => {
      const user = userEvent.setup();
      render(<Harness data={twoUsers} rowCount={2} selectionEnabled />);

      await user.click(screen.getByTestId("datatable-select-all"));
      expect(screen.getByTestId("selected-ids")).toHaveTextContent("user-1,user-2");

      await user.click(screen.getByTestId("datatable-select-all"));
      expect(screen.getByTestId("selected-ids")).toBeEmptyDOMElement();
    });

    it("shows an indeterminate header while only part of the page is selected", async () => {
      const user = userEvent.setup();
      render(<Harness data={twoUsers} rowCount={2} selectionEnabled />);

      await user.click(screen.getByTestId("datatable-select-row-user-1"));

      expect(screen.getByTestId("datatable-select-all")).toHaveAttribute("aria-checked", "mixed");
    });
  });

  it("renders the empty state when there are no users", () => {
    render(<Harness data={[]} rowCount={0} />);

    expect(screen.getByText("No users found")).toBeInTheDocument();
  });

  it("shows skeleton rows on the initial load instead of the empty state", () => {
    render(<Harness data={[]} rowCount={0} isLoading />);

    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("No users found")).not.toBeInTheDocument();
  });

  it("keeps the row menu out of the identity cell so only the name and menu act on a row", () => {
    render(<Harness />);

    const rows = screen.getAllByRole("row");
    const dataRow = rows[rows.length - 1];
    expect(within(dataRow).getByTestId("user-actions-user-1")).toBeInTheDocument();
  });
});
