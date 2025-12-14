import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { UserDataTable } from "./table";

const defaultFilters = {
  email: "",
  user_id: "",
  user_role: "",
  sso_user_id: "",
  team: "",
  model: "",
  min_spend: null,
  max_spend: null,
  sort_by: "",
  sort_order: "asc" as const,
};

const getDefaultProps = () => ({
  data: [] as any[],
  columns: [] as any[],
  accessToken: null,
  userRole: "Admin",
  possibleUIRoles: null as Record<string, Record<string, string>> | null,
  filters: defaultFilters,
  updateFilters: vi.fn(),
  initialFilters: defaultFilters,
  teams: [] as any[],
  handleEdit: vi.fn(),
  handleDelete: vi.fn(),
  handleResetPassword: vi.fn(),
  userListResponse: { users: [], total: 0, page: 1, page_size: 25, total_pages: 1 },
  currentPage: 1,
  handlePageChange: vi.fn(),
});

describe("UserDataTable", () => {
  it("should render the UserDataTable component", () => {
    render(<UserDataTable {...getDefaultProps()} />);

    expect(screen.getByText("Filters")).toBeInTheDocument();
  });

  it("should call onSortChange when clicking a sortable header", () => {
    const filters = {
      ...defaultFilters,
      sort_by: "created_at",
      sort_order: "desc" as const,
    };

    const onSortChange = vi.fn();

    const possibleUIRoles = {
      admin: { ui_label: "Admin" },
      user: { ui_label: "User" },
    };

    render(
      <UserDataTable
        {...getDefaultProps()}
        possibleUIRoles={possibleUIRoles}
        filters={filters}
        initialFilters={filters}
        onSortChange={onSortChange}
        currentSort={{ sortBy: filters.sort_by, sortOrder: filters.sort_order }}
      />,
    );

    const emailHeader = screen.getByRole("columnheader", { name: /email/i });
    act(() => {
      fireEvent.click(emailHeader);
    });

    expect(onSortChange).toHaveBeenCalledWith("user_email", "desc");
  });

  it("should show skeleton loaders when isLoading is true", () => {
    render(<UserDataTable {...getDefaultProps()} isLoading={true} />);

    expect(screen.queryByText(/Showing/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Previous/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Next/i })).not.toBeInTheDocument();
  });

  it("should show actual content when isLoading is false", () => {
    render(<UserDataTable {...getDefaultProps()} isLoading={false} />);

    expect(screen.getByText(/Showing/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Previous/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Next/i })).toBeInTheDocument();
  });

  it("should render all column headers", () => {
    const possibleUIRoles = {
      admin: { ui_label: "Admin" },
      user: { ui_label: "User" },
    };

    render(<UserDataTable {...getDefaultProps()} possibleUIRoles={possibleUIRoles} />);

    [
      "User ID",
      "Email",
      "Global Proxy Role",
      "User Alias",
      "Spend (USD)",
      "Budget (USD)",
      "SSO ID",
      "Virtual Keys",
      "Created At",
      "Updated At",
      "Actions",
    ].forEach((header) => {
      expect(screen.getByRole("columnheader", { name: header })).toBeInTheDocument();
    });
  });
});
