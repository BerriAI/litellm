/* @vitest-environment jsdom */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../../../tests/test-utils";
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
}));

// The detail view has its own test; stub it so this file covers the parent's swap.
vi.mock("./view_users/user_info_view", () => ({
  default: function UserInfoViewMock({ userId, onClose }: { userId: string; onClose: () => void }) {
    return (
      <div data-testid="user-info-view">
        {`detail:${userId}`}
        <button type="button" onClick={onClose}>
          Back to Users
        </button>
      </div>
    );
  },
}));

vi.mock("./default-user-settings/DefaultUserSettingsForm", () => ({
  DefaultUserSettingsForm: function DefaultUserSettingsFormMock() {
    const [value, setValue] = React.useState("");
    return (
      <section aria-label="Default user settings panel">
        <label>
          Default setting
          <input value={value} onChange={(event) => setValue(event.target.value)} />
        </label>
      </section>
    );
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

const manyUsers = {
  users: [makeUser("user-1", "test@example.com")],
  total: 500,
  page: 1,
  page_size: 25,
  total_pages: 20,
};

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

interface UrlOptions {
  searchParams?: string;
  onUrlUpdate?: OnUrlUpdateFunction;
}

const dashboard = (overrides: Partial<typeof defaultProps>) => (
  <QueryClientProvider client={createQueryClient()}>
    <ViewUserDashboard {...defaultProps} {...overrides} />
  </QueryClientProvider>
);

const renderDashboard = (overrides: Partial<typeof defaultProps> = {}, urlOptions: UrlOptions = {}) =>
  renderWithProviders(dashboard(overrides), urlOptions);

const renderDashboardKeepingMountWrites = (overrides: Partial<typeof defaultProps>, urlOptions: UrlOptions) =>
  render(
    <NuqsTestingAdapter
      searchParams={urlOptions.searchParams}
      onUrlUpdate={urlOptions.onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={false}
    >
      {dashboard(overrides)}
    </NuqsTestingAdapter>,
  );

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

const lastUserListArgs = (): unknown[] => userListCall.mock.lastCall ?? [];

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

  it("switches between the users table and default settings tabs for proxy admins", async () => {
    const user = userEvent.setup();
    renderDashboard();

    expect(await screen.findByText("test@example.com")).toBeInTheDocument();

    const usersTab = screen.getByRole("tab", { name: "Users" });
    const settingsTab = screen.getByRole("tab", { name: "Default User Settings" });
    expect(usersTab).toHaveAttribute("aria-selected", "true");

    await user.click(settingsTab);

    expect(settingsTab).toHaveAttribute("aria-selected", "true");
    expect(usersTab).toHaveAttribute("aria-selected", "false");
    expect(screen.getByRole("region", { name: "Default user settings panel" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: "Default setting" }), { target: { value: "unsaved change" } });

    await user.click(usersTab);

    expect(usersTab).toHaveAttribute("aria-selected", "true");
    expect(settingsTab).toHaveAttribute("aria-selected", "false");

    await user.click(settingsTab);

    expect(screen.getByRole("textbox", { name: "Default setting" })).toHaveValue("unsaved change");
  });

  it("renders invite and bulk invite as toolbar actions alongside the other admin controls", async () => {
    renderDashboard();

    const inviteButton = await screen.findByRole("button", { name: /\+ invite user/i });
    const bulkInviteButton = screen.getByRole("button", { name: /\+ bulk invite users/i });
    const toolbar = screen.getByTestId("toggle-user-selection").parentElement;

    expect(inviteButton.parentElement).toBe(toolbar);
    expect(bulkInviteButton.parentElement).toBe(toolbar);
  });

  it("shows the users table without admin controls for non-proxy admins", async () => {
    renderDashboard({ userRole: "Internal User" });

    expect(await screen.findByText("test@example.com")).toBeInTheDocument();
    expect(screen.queryByRole("tab")).not.toBeInTheDocument();
    expect(screen.queryByTestId("toggle-user-selection")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /\+ invite user/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /\+ bulk invite users/i })).not.toBeInTheDocument();
  });

  it("keeps actions unavailable while the user list is loading", () => {
    userListCall.mockReturnValue(new Promise(() => undefined));

    renderDashboard();

    expect(screen.getByText("Loading users…")).toBeInTheDocument();
    expect(screen.queryByTestId("toggle-user-selection")).not.toBeInTheDocument();
    expect(screen.queryByTestId("bulk-edit-users")).not.toBeInTheDocument();
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

  describe("user detail in the URL", () => {
    it("swaps to the detail view and pushes only the user when the identity cell is clicked", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderDashboard({}, { searchParams: "?user_tab=details&edit=true", onUrlUpdate });

      await waitFor(() => {
        expect(screen.getByText("test@example.com")).toBeInTheDocument();
      });

      await user.click(screen.getByRole("button", { name: /user-1/ }));

      expect(await screen.findByTestId("user-info-view")).toHaveTextContent("detail:user-1");
      expect(screen.queryByText("test@example.com")).not.toBeInTheDocument();
      expect(onUrlUpdate).toHaveBeenCalledTimes(1);
      const [update] = onUrlUpdate.mock.calls[0];
      expect(update.searchParams.get("user")).toBe("user-1");
      expect(update.searchParams.has("user_tab")).toBe(false);
      expect(update.searchParams.has("edit")).toBe(false);
      expect(update.options.history).toBe("push");
    });

    it("opens the Details tab in edit mode with one URL write from the row actions menu", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderDashboard({}, { onUrlUpdate });

      await waitFor(() => {
        expect(screen.getByText("test@example.com")).toBeInTheDocument();
      });

      await user.click(screen.getByTestId("user-actions-user-1"));
      await user.click(await screen.findByTestId("user-action-edit"));

      expect(await screen.findByTestId("user-info-view")).toHaveTextContent("detail:user-1");
      expect(onUrlUpdate).toHaveBeenCalledTimes(1);
      const [update] = onUrlUpdate.mock.calls[0];
      expect(update.searchParams.get("user")).toBe("user-1");
      expect(update.searchParams.get("user_tab")).toBe("details");
      expect(update.searchParams.get("edit")).toBe("true");
      expect(update.options.history).toBe("push");
    });

    it("opens the detail view for a ?user= deep link", async () => {
      renderDashboard({}, { searchParams: "?user=user-9" });

      expect(await screen.findByTestId("user-info-view")).toHaveTextContent("detail:user-9");
      expect(screen.queryByTestId("datatable-search")).not.toBeInTheDocument();
    });

    it("drops the user, detail tab and edit flag in one pushed entry when the detail view closes", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderDashboard({}, { searchParams: "?user=user-9&user_tab=details&edit=true&filter_team=t-1", onUrlUpdate });

      await user.click(await screen.findByRole("button", { name: "Back to Users" }));

      expect(await screen.findByText("test@example.com")).toBeInTheDocument();
      expect(onUrlUpdate).toHaveBeenCalledTimes(1);
      const [update] = onUrlUpdate.mock.calls[0];
      expect(update.searchParams.has("user")).toBe(false);
      expect(update.searchParams.has("user_tab")).toBe(false);
      expect(update.searchParams.has("edit")).toBe(false);
      expect(update.searchParams.get("filter_team")).toBe("t-1");
      expect(update.options.history).toBe("push");
    });
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
      expect(screen.getByTestId("bulk-edit-users")).toBeEnabled();

      await user.click(screen.getByTestId("datatable-select-all"));
      expect(screen.getByTestId("bulk-edit-users")).toHaveTextContent("Bulk Edit (2 selected)");
    });

    it("clears the selection when the sort changes", async () => {
      const user = userEvent.setup();
      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText("ada@example.com")).toBeInTheDocument();
      });

      await user.click(screen.getByTestId("toggle-user-selection"));
      await user.click(screen.getByTestId("datatable-select-row-user-1"));
      expect(screen.getByTestId("bulk-edit-users")).toHaveTextContent("Bulk Edit (1 selected)");

      await user.click(screen.getByTestId("sort-header-user_email"));

      await waitFor(() => {
        expect(screen.getByTestId("bulk-edit-users")).toHaveTextContent("Bulk Edit (0 selected)");
      });
    });

    it("clears the selection when the page changes", async () => {
      userListCall.mockResolvedValue({ ...manyUsers, users: [makeUser("user-1", "ada@example.com")] });
      const user = userEvent.setup();
      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText("ada@example.com")).toBeInTheDocument();
      });

      await user.click(screen.getByTestId("toggle-user-selection"));
      await user.click(screen.getByTestId("datatable-select-row-user-1"));
      expect(screen.getByTestId("bulk-edit-users")).toHaveTextContent("Bulk Edit (1 selected)");

      await user.click(screen.getByTestId("pagination-next"));

      await waitFor(() => {
        expect(screen.getByTestId("bulk-edit-users")).toHaveTextContent("Bulk Edit (0 selected)");
      });
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

    it("sends the toolbar search as the combined search param instead of user_email", async () => {
      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText("test@example.com")).toBeInTheDocument();
      });

      const searchedUserId = "a6f5c02b-0163-45ce-815f-f88d10e95686";
      fireEvent.change(screen.getByPlaceholderText("Search by email or ID…"), { target: { value: searchedUserId } });

      await waitFor(() => {
        const latest = userListCall.mock.calls[userListCall.mock.calls.length - 1];
        expect(latest[11]).toBe(searchedUserId);
      });
      const latest = userListCall.mock.calls[userListCall.mock.calls.length - 1];
      expect(latest[1]).toBeNull();
      expect(latest[4]).toBeNull();
      expect(latest[2]).toBe(1);
    });

    it("replaces the previous rows with the loading state while the search request is pending", async () => {
      renderDashboard();
      expect(await screen.findByText("test@example.com")).toBeInTheDocument();

      userListCall.mockReturnValue(new Promise(() => undefined));
      fireEvent.change(screen.getByPlaceholderText("Search by email or ID…"), { target: { value: "zzznomatch" } });

      expect(await screen.findByText("Loading users…")).toBeInTheDocument();
      expect(screen.queryByText("test@example.com")).not.toBeInTheDocument();
    });
  });

  describe("table state in the URL", () => {
    it("reads search, sort, page, page size and every filter from the URL into the user list request", async () => {
      userListCall.mockResolvedValue(manyUsers);
      renderDashboard(
        {},
        {
          searchParams:
            "?user_search=ada&sort_by=spend&sort_order=asc&page=3&page_size=50" +
            "&filter_user_id=u-1&filter_sso_id=sso-1&filter_role=User&filter_team=team-1",
        },
      );

      await waitFor(() => expect(userListCall).toHaveBeenCalled());
      expect(lastUserListArgs()).toEqual([
        "test-token",
        ["u-1"],
        3,
        50,
        null,
        "User",
        "team-1",
        "sso-1",
        "spend",
        "asc",
        null,
        "ada",
      ]);
      expect(screen.getByPlaceholderText("Search by email or ID…")).toHaveValue("ada");
      expect(screen.getByTestId("filter-chip-user_role")).toHaveTextContent("User");
      expect(screen.getByTestId("filter-chip-sso_user_id")).toHaveTextContent("sso-1");
      expect(await screen.findByTestId("pagination-page")).toHaveTextContent("Page 3 of 10");
    });

    it("falls back to created_at for a sort_by that is not a sortable column", async () => {
      renderDashboard({}, { searchParams: "?sort_by=user_alias&sort_order=asc" });

      await waitFor(() => expect(userListCall).toHaveBeenCalled());
      expect(lastUserListArgs()[8]).toBe("created_at");
      expect(lastUserListArgs()[9]).toBe("asc");
    });

    it("sends every sortable column header's id as the requested sort", async () => {
      const user = userEvent.setup();
      renderDashboard();
      await screen.findByText("test@example.com");

      const sortableIds = screen
        .getAllByTestId(/^sort-header-/)
        .map((header) => header.getAttribute("data-testid")?.replace("sort-header-", "") ?? "")
        .filter((columnId) => columnId !== "created_at");
      expect(sortableIds).toContain("user_email");

      for (const columnId of sortableIds) {
        await user.click(screen.getByTestId(`sort-header-${columnId}`));
        await waitFor(() => expect(lastUserListArgs()[8]).toBe(columnId));
      }
    });

    it("writes the sort to the URL and goes back to the first page", async () => {
      userListCall.mockResolvedValue(manyUsers);
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderDashboard({}, { searchParams: "?page=2", onUrlUpdate });
      await screen.findByText("test@example.com");

      await user.click(screen.getByTestId("sort-header-user_email"));

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("sort_by")).toBe("user_email"));
      expect(lastUrl(onUrlUpdate)?.get("sort_order")).toBe("asc");
      expect(lastUrl(onUrlUpdate)?.has("page")).toBe(false);
    });

    it("writes the search box to user_search and goes back to the first page", async () => {
      userListCall.mockResolvedValue(manyUsers);
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderDashboard({}, { searchParams: "?page=2", onUrlUpdate });
      await screen.findByText("test@example.com");

      fireEvent.change(screen.getByPlaceholderText("Search by email or ID…"), { target: { value: "grace" } });

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("user_search")).toBe("grace"));
      expect(lastUrl(onUrlUpdate)?.has("page")).toBe(false);
      expect(lastUrl(onUrlUpdate)?.has("search")).toBe(false);
    });

    it("writes applied drawer filters under their filter_ keys and requests them", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderDashboard({}, { onUrlUpdate });
      await screen.findByText("test@example.com");

      await user.click(screen.getByTestId("datatable-filters-trigger"));
      fireEvent.change(await screen.findByTestId("users-filter-user-id"), { target: { value: "u-7" } });
      fireEvent.change(screen.getByTestId("users-filter-sso-id"), { target: { value: "sso-7" } });
      await user.click(screen.getByTestId("filter-drawer-apply"));

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("filter_user_id")).toBe("u-7"));
      expect(lastUrl(onUrlUpdate)?.get("filter_sso_id")).toBe("sso-7");
      expect(lastUrl(onUrlUpdate)?.has("filter_sso_user_id")).toBe(false);
      await waitFor(() => {
        expect(lastUserListArgs()[1]).toEqual(["u-7"]);
        expect(lastUserListArgs()[7]).toBe("sso-7");
      });
    });

    it("writes the next page to the URL and requests it", async () => {
      userListCall.mockResolvedValue(manyUsers);
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderDashboard({}, { onUrlUpdate });
      await screen.findByText("test@example.com");

      await user.click(screen.getByTestId("pagination-next"));

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("page")).toBe("2"));
      await waitFor(() => expect(lastUserListArgs()[2]).toBe(2));
    });

    it("keeps ?page= in the URL when the user list request fails", async () => {
      userListCall.mockRejectedValue(new Error("boom"));
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderDashboardKeepingMountWrites({}, { searchParams: "?page=3", onUrlUpdate });

      expect(await screen.findByText("No users found")).toBeInTheDocument();
      await new Promise((resolve) => setTimeout(resolve, 100));

      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 3");
      expect(onUrlUpdate.mock.calls.every(([update]) => update.searchParams.get("page") === "3")).toBe(true);
      expect(userListCall.mock.calls.every((args) => args[2] === 3)).toBe(true);
    });
  });

  describe("top-level tab in the URL", () => {
    it("opens the Default User Settings tab named by ?tab=", async () => {
      renderDashboard({}, { searchParams: "?tab=default-settings" });

      expect(await screen.findByRole("tab", { name: "Default User Settings" })).toHaveAttribute(
        "aria-selected",
        "true",
      );
      expect(screen.getByRole("region", { name: "Default user settings panel" })).toBeVisible();
    });

    it("writes the chosen tab to the URL and drops it again for the users default", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderDashboard({}, { onUrlUpdate });

      await user.click(await screen.findByRole("tab", { name: "Default User Settings" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("tab")).toBe("default-settings"));

      await user.click(screen.getByRole("tab", { name: "Users" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.has("tab")).toBe(false));
    });

    it("drops ?tab=default-settings for a non-admin, who has no settings tab", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderDashboardKeepingMountWrites(
        { userRole: "Internal User" },
        { searchParams: "?tab=default-settings&user_search=ada", onUrlUpdate },
      );

      expect(await screen.findByText("test@example.com")).toBeInTheDocument();
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.has("tab")).toBe(false));
      expect(lastUrl(onUrlUpdate)?.get("user_search")).toBe("ada");
    });
  });
});
