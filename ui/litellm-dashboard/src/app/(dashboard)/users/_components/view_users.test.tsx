/* @vitest-environment jsdom */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ViewUserDashboard from "./view_users";

const userListCall = vi.fn();

// Mock the networking module
vi.mock("@/components/networking", () => ({
  userListCall: (...args: unknown[]) => userListCall(...args),
  userDeleteCall: vi.fn().mockResolvedValue({}),
  getPossibleUserRoles: vi.fn().mockResolvedValue({
    Admin: { ui_label: "Admin" },
    User: { ui_label: "User" },
  }),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
  invitationCreateCall: vi.fn().mockResolvedValue({}),
  userUpdateUserCall: vi.fn().mockResolvedValue({}),
  getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost:4000"),
  getProxyUISettings: vi.fn().mockResolvedValue({
    PROXY_BASE_URL: null,
    PROXY_LOGOUT_URL: null,
    DEFAULT_TEAM_DISABLED: false,
    SSO_ENABLED: false,
  }),
  getInternalUserSettings: vi.fn().mockResolvedValue({}),
}));

// The detail view has its own test; stub it so this file covers the parent's swap.
vi.mock("./view_users/user_info_view", () => ({
  default: function UserInfoViewMock({ userId, startInEditMode }: { userId: string; startInEditMode?: boolean }) {
    return <div data-testid="user-info-view">{`detail:${userId}:${String(Boolean(startInEditMode))}`}</div>;
  },
}));

// Mock NotificationsManager
vi.mock("@/components/molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

const makeUser = (userId: string, email: string) => ({
  user_id: userId,
  user_email: email,
  user_alias: null,
  user_role: "Admin",
  spend: 100.5,
  max_budget: null,
  models: [],
  key_count: 2,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  sso_user_id: null,
  budget_duration: null,
});

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

const defaultProps = {
  accessToken: "test-token",
  token: "test-token",
  userRole: "Admin",
  userID: "admin-user-id",
  teams: [],
};

const renderDashboard = () =>
  render(
    <QueryClientProvider client={createQueryClient()}>
      <ViewUserDashboard {...defaultProps} />
    </QueryClientProvider>,
  );

describe("ViewUserDashboard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    userListCall.mockResolvedValue({
      users: [makeUser("user-1", "test@example.com")],
      total: 1,
      page: 1,
      page_size: 25,
      total_pages: 1,
    });
  });

  it("should render the ViewUserDashboard component", async () => {
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText("Users")).toBeInTheDocument();
    });

    expect(screen.getAllByText("Default User Settings").length).toBeGreaterThan(0);
  });

  it("should show delete modal after choosing delete from the row actions menu", async () => {
    const user = userEvent.setup();
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText("test@example.com")).toBeInTheDocument();
    });

    expect(screen.queryByText("Delete User?")).not.toBeInTheDocument();

    await user.click(screen.getByTestId("user-actions-user-1"));
    await user.click(await screen.findByTestId("user-action-delete"));

    await waitFor(() => {
      expect(screen.getByText("Delete User?")).toBeInTheDocument();
    });
    expect(
      screen.getByText("Are you sure you want to delete this user? This action cannot be undone."),
    ).toBeInTheDocument();
    expect(screen.getAllByText("user-1").length).toBeGreaterThan(0);
  });

  it("should swap to the detail view when the identity cell is clicked", async () => {
    const user = userEvent.setup();
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText("test@example.com")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /user-1/ }));

    expect(await screen.findByTestId("user-info-view")).toHaveTextContent("detail:user-1:false");
    expect(screen.queryByText("test@example.com")).not.toBeInTheDocument();
  });

  it("should open the detail view in edit mode from the row actions menu", async () => {
    const user = userEvent.setup();
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText("test@example.com")).toBeInTheDocument();
    });

    await user.click(screen.getByTestId("user-actions-user-1"));
    await user.click(await screen.findByTestId("user-action-edit"));

    expect(await screen.findByTestId("user-info-view")).toHaveTextContent("detail:user-1:true");
  });

  describe("bulk edit selection", () => {
    beforeEach(() => {
      userListCall.mockResolvedValue({
        users: [makeUser("user-1", "ada@example.com"), makeUser("user-2", "grace@example.com")],
        total: 2,
        page: 1,
        page_size: 25,
        total_pages: 1,
      });
    });

    it("reveals selection checkboxes only while selection mode is on", async () => {
      const user = userEvent.setup();
      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText("ada@example.com")).toBeInTheDocument();
      });

      expect(screen.queryByTestId("datatable-select-all")).not.toBeInTheDocument();

      await user.click(screen.getByTestId("toggle-user-selection"));
      expect(screen.getByTestId("datatable-select-all")).toBeInTheDocument();

      await user.click(screen.getByTestId("toggle-user-selection"));
      expect(screen.queryByTestId("datatable-select-all")).not.toBeInTheDocument();
    });

    it("counts the selected rows in the bulk edit button and enables it once a row is picked", async () => {
      const user = userEvent.setup();
      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText("ada@example.com")).toBeInTheDocument();
      });

      await user.click(screen.getByTestId("toggle-user-selection"));

      const bulkEdit = screen.getByTestId("bulk-edit-users");
      expect(bulkEdit).toHaveTextContent("Bulk Edit (0 selected)");
      expect(bulkEdit).toBeDisabled();

      await user.click(screen.getByTestId("datatable-select-row-user-2"));
      expect(screen.getByTestId("bulk-edit-users")).toHaveTextContent("Bulk Edit (1 selected)");
      expect(screen.getByTestId("bulk-edit-users")).not.toBeDisabled();

      await user.click(screen.getByTestId("datatable-select-all"));
      expect(screen.getByTestId("bulk-edit-users")).toHaveTextContent("Bulk Edit (2 selected)");
    });

    it("clears the selection when selection mode is cancelled", async () => {
      const user = userEvent.setup();
      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText("ada@example.com")).toBeInTheDocument();
      });

      await user.click(screen.getByTestId("toggle-user-selection"));
      await user.click(screen.getByTestId("datatable-select-row-user-1"));
      expect(screen.getByTestId("bulk-edit-users")).toHaveTextContent("Bulk Edit (1 selected)");

      await user.click(screen.getByTestId("toggle-user-selection"));
      await user.click(screen.getByTestId("toggle-user-selection"));

      expect(screen.getByTestId("bulk-edit-users")).toHaveTextContent("Bulk Edit (0 selected)");
    });
  });

  describe("server-side query wiring", () => {
    it("requests page 1 with the default created_at desc sort", async () => {
      renderDashboard();

      await waitFor(() => {
        expect(userListCall).toHaveBeenCalled();
      });

      const [, userIds, page, pageSize, , , , , sortBy, sortOrder] = userListCall.mock.calls[0];
      expect(userIds).toBeNull();
      expect(page).toBe(1);
      expect(pageSize).toBe(25);
      expect(sortBy).toBe("created_at");
      expect(sortOrder).toBe("desc");
    });

    it("sends the clicked column as sort_by and resets to the first page", async () => {
      const user = userEvent.setup();
      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText("test@example.com")).toBeInTheDocument();
      });

      await user.click(screen.getByTestId("sort-header-user_email"));

      await waitFor(() => {
        const latest = userListCall.mock.calls[userListCall.mock.calls.length - 1];
        expect(latest[8]).toBe("user_email");
        expect(latest[9]).toBe("asc");
        expect(latest[2]).toBe(1);
      });
    });
  });
});
