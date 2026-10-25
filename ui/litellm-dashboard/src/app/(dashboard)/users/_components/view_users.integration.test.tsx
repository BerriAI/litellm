/* @vitest-environment jsdom */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../../../tests/test-utils";
import ViewUserDashboard from "./view_users";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";
import {
  invitationCreateCall,
  userDeleteCall,
  modelAvailableCall,
  getPossibleUserRoles,
} from "@/components/networking";

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
  getUiPageUrl: (url: string) => url,
  getProxyUISettings: vi.fn().mockResolvedValue({
    PROXY_BASE_URL: null,
    PROXY_LOGOUT_URL: null,
    DEFAULT_TEAM_DISABLED: false,
    SSO_ENABLED: false,
  }),
}));

// The detail view has its own test; stub it so this file covers the parent's swap.
vi.mock("./view_users/user_info_view", () => ({
  default: function UserInfoViewMock({ userId, startInEditMode }: { userId: string; startInEditMode?: boolean }) {
    return <div data-testid="user-info-view">{`detail:${userId}:${String(Boolean(startInEditMode))}`}</div>;
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

const renderDashboard = (overrides: Partial<typeof defaultProps> = {}) =>
  renderWithProviders(
    <QueryClientProvider client={createQueryClient()}>
      <ViewUserDashboard {...defaultProps} {...overrides} />
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
      expect(screen.getByTestId("bulk-edit-users")).toBeEnabled();

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

    it.each([
      ["user_alias", "asc"],
      ["max_budget", "desc"],
      ["updated_at", "asc"],
      ["status", "asc"],
      ["key_count", "desc"],
      ["sso_user_id", "asc"],
    ])("sorts %s through the API and resets pagination", async (field, direction) => {
      const user = userEvent.setup();
      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText("test@example.com")).toBeInTheDocument();
      });

      await user.click(screen.getByTestId(`sort-header-${field}`));

      await waitFor(() => {
        const latest = userListCall.mock.calls[userListCall.mock.calls.length - 1];
        expect(latest[8]).toBe(field);
        expect(latest[9]).toBe(direction);
        expect(latest[2]).toBe(1);
      });
    });

    it("sends the toolbar search as the combined search param instead of user_email", async () => {
      const user = userEvent.setup();
      renderDashboard();

      await waitFor(() => {
        expect(screen.getByText("test@example.com")).toBeInTheDocument();
      });

      const searchedUserId = "a6f5c02b-0163-45ce-815f-f88d10e95686";
      await user.type(screen.getByPlaceholderText("Search by email or ID…"), searchedUserId);

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
});

describe("用户列表异步通知的语言边界", () => {
  beforeEach(async () => {
    localStorage.clear();
    vi.clearAllMocks();
    const userList = {
      users: [makeUser("Home", "home@example.com")],
      total: 1,
      page: 1,
      page_size: 25,
      total_pages: 1,
    };
    userListCall.mockResolvedValue(userList);
    const actual = await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast");
    // 固定通知交给真实语言上下文和 Sonner，保留本文件既有的详情页替身。
    vi.mocked(toast.success).mockImplementation(actual.toast.success);
    vi.mocked(toast.fromError).mockImplementation(actual.toast.fromError);
  });

  afterEach(async () => {
    const actual = await vi.importActual<typeof import("@/lib/toast")>("@/lib/toast");
    act(() => actual.toast.dismiss());
    vi.mocked(toast.success).mockReset();
    vi.mocked(toast.fromError).mockReset();
  });

  const renderLocalizedDashboard = () => {
    const client = createQueryClient();
    const tree = (defaultLanguage: string) => (
      <QueryClientProvider client={client}>
        <LanguageProvider defaultLanguage={defaultLanguage} userId="users-review" storageScope="/users">
          <LanguageSelector />
          <ViewUserDashboard {...defaultProps} />
          <Toaster expand duration={Infinity} />
        </LanguageProvider>
      </QueryClientProvider>
    );
    return { ...renderWithProviders(tree("en")), tree };
  };

  const switchLanguage = (value: "en" | "zh-CN") =>
    fireEvent.change(screen.getByRole("combobox", { name: /Interface language|界面语言/ }), { target: { value } });

  const chooseUserAction = async (user: ReturnType<typeof userEvent.setup>, action: "reset-password" | "delete") => {
    expect(await screen.findByText("home@example.com")).toBeInTheDocument();
    await user.click(screen.getByTestId("user-actions-Home"));
    await user.click(await screen.findByTestId(`user-action-${action}`));
  };

  it("生成密码重置链接的等待提示跟随语言，请求不会重复发送", async () => {
    const user = userEvent.setup();
    const pending = Promise.withResolvers<Awaited<ReturnType<typeof invitationCreateCall>>>();
    vi.mocked(invitationCreateCall).mockReturnValueOnce(pending.promise);
    renderLocalizedDashboard();
    await chooseUserAction(user, "reset-password");
    expect(await screen.findByText("Generating password reset link...")).toBeInTheDocument();
    const modelLoads = vi.mocked(modelAvailableCall).mock.calls.length;
    const roleLoads = vi.mocked(getPossibleUserRoles).mock.calls.length;

    switchLanguage("zh-CN");
    expect(screen.getByText("正在生成密码重置链接...")).toBeInTheDocument();
    switchLanguage("en");
    expect(screen.getByText("Generating password reset link...")).toBeInTheDocument();
    expect(invitationCreateCall).toHaveBeenCalledExactlyOnceWith("test-token", "Home");
    expect(userListCall).toHaveBeenCalledTimes(1);
    expect(modelAvailableCall).toHaveBeenCalledTimes(modelLoads);
    expect(getPossibleUserRoles).toHaveBeenCalledTimes(roleLoads);
    await act(async () => pending.reject(new Error("private-reset-detail")));
  });

  it.each([false, true])("密码重置失败提示只显示当前语言的本地说明，等待中切换=%s", async (switchWhilePending) => {
    const user = userEvent.setup();
    const pending = Promise.withResolvers<Awaited<ReturnType<typeof invitationCreateCall>>>();
    vi.mocked(invitationCreateCall).mockReturnValueOnce(pending.promise);
    renderLocalizedDashboard();
    await chooseUserAction(user, "reset-password");
    if (switchWhilePending) switchLanguage("zh-CN");
    await act(async () => pending.reject(new Error("private-reset-detail")));
    if (!switchWhilePending) {
      expect(await screen.findByText("Failed to generate password reset link")).toBeInTheDocument();
      switchLanguage("zh-CN");
    }

    expect(await screen.findByText("生成密码重置链接失败，请重试")).toBeInTheDocument();
    expect(screen.queryByText("private-reset-detail")).not.toBeInTheDocument();
    switchLanguage("en");
    expect(screen.getByText("Failed to generate password reset link")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(invitationCreateCall).toHaveBeenCalledExactlyOnceWith("test-token", "Home");
    expect(userListCall).toHaveBeenCalledTimes(1);
  });

  it.each([false, true])("删除用户的结果提示跟随当前语言，确认和列表更新不变，失败=%s", async (failed) => {
    const user = userEvent.setup();
    const pending = Promise.withResolvers<Awaited<ReturnType<typeof userDeleteCall>>>();
    vi.mocked(userDeleteCall).mockReturnValueOnce(pending.promise);
    const view = renderLocalizedDashboard();
    await chooseUserAction(user, "delete");
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveTextContent("Home");
    expect(userDeleteCall).not.toHaveBeenCalled();
    await user.click(within(dialog).getByRole("button", { name: "Delete" }));

    // 删除确认框开启时模拟全站默认更新，不点击模态外控件。
    view.rerender(view.tree("zh-CN"));
    expect(within(dialog).getByRole("button", { name: "正在删除…" })).toBeDisabled();
    expect(userDeleteCall).toHaveBeenCalledExactlyOnceWith("test-token", ["Home"]);
    await act(async () => {
      if (failed) pending.reject(new Error("private-delete-detail"));
      else pending.resolve({});
    });

    expect(await screen.findByText(failed ? "删除用户失败，请重试" : "用户已删除")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.queryByText("private-delete-detail")).not.toBeInTheDocument();
    if (failed) expect(screen.getByText("home@example.com")).toBeInTheDocument();
    else expect(screen.queryByText("home@example.com")).not.toBeInTheDocument();
    switchLanguage("en");
    expect(screen.getByText(failed ? "Failed to delete user" : "User deleted successfully")).toBeInTheDocument();
    expect(userDeleteCall).toHaveBeenCalledTimes(1);
    expect(userListCall).toHaveBeenCalledTimes(1);
  });
});
