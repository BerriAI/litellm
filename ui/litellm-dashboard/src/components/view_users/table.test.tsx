import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import React from "react";

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

    const { getByText } = render(
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

    expect(getByText("Filters")).toBeInTheDocument();
  });
});
