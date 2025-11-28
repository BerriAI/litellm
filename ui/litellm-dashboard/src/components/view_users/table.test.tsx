import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { UserDataTable } from "./table";

describe("UserDataTable", () => {
  it("should render the UserDataTable component", () => {
    const filters = {
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

    const updateFilters = vi.fn();

    render(
      <UserDataTable
        data={[]}
        columns={[]}
        accessToken={null}
        userRole={"Admin"}
        possibleUIRoles={null}
        filters={filters}
        updateFilters={updateFilters}
        initialFilters={filters}
        teams={[]}
        handleEdit={vi.fn()}
        handleDelete={vi.fn()}
        handleResetPassword={vi.fn()}
        userListResponse={{ users: [], total: 0, page: 1, page_size: 25, total_pages: 1 }}
        currentPage={1}
        handlePageChange={vi.fn()}
      />,
    );

    expect(screen.getByText("Filters")).toBeInTheDocument();
  });

  it("should call onSortChange when clicking a sortable header", () => {
    const filters = {
      email: "",
      user_id: "",
      user_role: "",
      sso_user_id: "",
      team: "",
      model: "",
      min_spend: null,
      max_spend: null,
      sort_by: "created_at",
      sort_order: "desc" as const,
    };

    const updateFilters = vi.fn();
    const onSortChange = vi.fn();

    const possibleUIRoles = {
      admin: { ui_label: "Admin" },
      user: { ui_label: "User" },
    };

    render(
      <UserDataTable
        data={[]}
        columns={[]}
        accessToken={null}
        userRole={"Admin"}
        possibleUIRoles={possibleUIRoles}
        filters={filters}
        updateFilters={updateFilters}
        initialFilters={filters}
        teams={[]}
        handleEdit={vi.fn()}
        handleDelete={vi.fn()}
        handleResetPassword={vi.fn()}
        userListResponse={{ users: [], total: 0, page: 1, page_size: 25, total_pages: 1 }}
        currentPage={1}
        handlePageChange={vi.fn()}
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
    const filters = {
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

    const updateFilters = vi.fn();

    render(
      <UserDataTable
        data={[]}
        columns={[]}
        accessToken={null}
        userRole={"Admin"}
        possibleUIRoles={null}
        filters={filters}
        updateFilters={updateFilters}
        initialFilters={filters}
        teams={[]}
        handleEdit={vi.fn()}
        handleDelete={vi.fn()}
        handleResetPassword={vi.fn()}
        userListResponse={{ users: [], total: 0, page: 1, page_size: 25, total_pages: 1 }}
        currentPage={1}
        handlePageChange={vi.fn()}
        isLoading={true}
      />,
    );

    expect(screen.queryByText(/Showing/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Previous/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Next/i })).not.toBeInTheDocument();
  });

  it("should show actual content when isLoading is false", () => {
    const filters = {
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

    const updateFilters = vi.fn();

    render(
      <UserDataTable
        data={[]}
        columns={[]}
        accessToken={null}
        userRole={"Admin"}
        possibleUIRoles={null}
        filters={filters}
        updateFilters={updateFilters}
        initialFilters={filters}
        teams={[]}
        handleEdit={vi.fn()}
        handleDelete={vi.fn()}
        handleResetPassword={vi.fn()}
        userListResponse={{ users: [], total: 0, page: 1, page_size: 25, total_pages: 1 }}
        currentPage={1}
        handlePageChange={vi.fn()}
        isLoading={false}
      />,
    );

    expect(screen.getByText(/Showing/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Previous/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Next/i })).toBeInTheDocument();
  });
});
