import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../tests/test-utils";
import BulkEditUserModal from "./BulkEditUsers";
import { userBulkUpdateUserCall, teamBulkMemberAddCall } from "./networking";
import NotificationsManager from "./molecules/notifications_manager";

vi.mock("./networking", () => ({
  userBulkUpdateUserCall: vi.fn(),
  teamBulkMemberAddCall: vi.fn(),
}));

vi.mock("./user_edit_view", () => ({
  UserEditView: ({ onSubmit, onCancel }: { onSubmit: (values: any) => void; onCancel: () => void }) => (
    <div data-testid="user-edit-view">
      <button onClick={() => onSubmit({ user_role: "admin", max_budget: 100 })}>Submit</button>
      <button onClick={onCancel}>Cancel</button>
    </div>
  ),
}));

const mockUserBulkUpdateUserCall = vi.mocked(userBulkUpdateUserCall);
const mockTeamBulkMemberAddCall = vi.mocked(teamBulkMemberAddCall);

const defaultProps = {
  open: true,
  onCancel: vi.fn(),
  selectedUsers: [
    { user_id: "user1", user_email: "user1@example.com", user_role: "user", max_budget: 50 },
    { user_id: "user2", user_email: "user2@example.com", user_role: "admin", max_budget: null },
  ],
  possibleUIRoles: {
    admin: { ui_label: "Admin", description: "Administrator role" },
    user: { ui_label: "User", description: "Regular user role" },
  },
  accessToken: "test-token",
  onSuccess: vi.fn(),
  teams: [
    { team_id: "team1", team_alias: "Team 1" },
    { team_id: "team2", team_alias: "Team 2" },
  ],
  userRole: "Admin",
  userModels: ["gpt-4", "gpt-3.5-turbo"],
  allowAllUsers: false,
};

describe("BulkEditUserModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUserBulkUpdateUserCall.mockResolvedValue({
      results: [],
      total_requested: 2,
      successful_updates: 2,
      failed_updates: 0,
    });
    mockTeamBulkMemberAddCall.mockResolvedValue({
      successful_additions: 2,
      failed_additions: 0,
    });
  });

  it("should render without crashing", () => {
    renderWithProviders(<BulkEditUserModal {...defaultProps} />);

    expect(screen.getByText(`Bulk Edit ${defaultProps.selectedUsers.length} User(s)`)).toBeInTheDocument();
  });

  it("should display modal title with correct user count", () => {
    renderWithProviders(<BulkEditUserModal {...defaultProps} />);

    expect(screen.getByText("Bulk Edit 2 User(s)")).toBeInTheDocument();
  });

  it("should display selected users table when modal is open", () => {
    renderWithProviders(<BulkEditUserModal {...defaultProps} />);

    expect(screen.getByText("Selected Users (2):")).toBeInTheDocument();
    expect(screen.getByText("user1")).toBeInTheDocument();
    expect(screen.getByText("user2")).toBeInTheDocument();
    expect(screen.getByText("user1@example.com")).toBeInTheDocument();
    expect(screen.getByText("user2@example.com")).toBeInTheDocument();
  });

  it("should display user roles in table", () => {
    renderWithProviders(<BulkEditUserModal {...defaultProps} />);

    expect(screen.getByText("User")).toBeInTheDocument();
    expect(screen.getByText("Admin")).toBeInTheDocument();
  });

  it("should display budget information in table", () => {
    renderWithProviders(<BulkEditUserModal {...defaultProps} />);

    expect(screen.getByText("$50")).toBeInTheDocument();
    expect(screen.getByText("Unlimited")).toBeInTheDocument();
  });

  it("should call onCancel when cancel button is clicked", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    renderWithProviders(<BulkEditUserModal {...defaultProps} onCancel={onCancel} />);

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    await user.click(cancelButton);

    expect(onCancel).toHaveBeenCalledTimes(1);
  });


  it("should show update all users checkbox when allowAllUsers is true", () => {
    renderWithProviders(<BulkEditUserModal {...defaultProps} allowAllUsers={true} />);

    expect(screen.getByRole("checkbox", { name: /update all users/i })).toBeInTheDocument();
  });

  it("should not show update all users checkbox when allowAllUsers is false", () => {
    renderWithProviders(<BulkEditUserModal {...defaultProps} allowAllUsers={false} />);

    expect(screen.queryByRole("checkbox", { name: /update all users/i })).not.toBeInTheDocument();
  });

  it("should toggle update all users mode", async () => {
    const user = userEvent.setup();
    renderWithProviders(<BulkEditUserModal {...defaultProps} allowAllUsers={true} />);

    const checkbox = screen.getByRole("checkbox", { name: /update all users/i });
    expect(checkbox).not.toBeChecked();

    await user.click(checkbox);

    expect(checkbox).toBeChecked();
    expect(screen.getByText("Bulk Edit All Users")).toBeInTheDocument();
  });

  it("should show warning message when update all users is enabled", async () => {
    const user = userEvent.setup();
    renderWithProviders(<BulkEditUserModal {...defaultProps} allowAllUsers={true} />);

    const checkbox = screen.getByRole("checkbox", { name: /update all users/i });
    await user.click(checkbox);

    expect(screen.getByText(/this will apply changes to all users/i)).toBeInTheDocument();
  });

  it("should hide selected users table when update all users is enabled", async () => {
    const user = userEvent.setup();
    renderWithProviders(<BulkEditUserModal {...defaultProps} allowAllUsers={true} />);

    expect(screen.getByText("Selected Users (2):")).toBeInTheDocument();

    const checkbox = screen.getByRole("checkbox", { name: /update all users/i });
    await user.click(checkbox);

    expect(screen.queryByText("Selected Users (2):")).not.toBeInTheDocument();
  });

  it("should display team management section", () => {
    renderWithProviders(<BulkEditUserModal {...defaultProps} />);

    expect(screen.getByText("Team Management")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /add selected users to teams/i })).toBeInTheDocument();
  });

  it("should show team budget input when add to teams is checked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<BulkEditUserModal {...defaultProps} />);

    const addToTeamsCheckbox = screen.getByRole("checkbox", { name: /add selected users to teams/i });
    await user.click(addToTeamsCheckbox);

    expect(screen.getByText("Team Budget (Optional):")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Max budget per user in team")).toBeInTheDocument();
  });

  it("should render UserEditView component", () => {
    renderWithProviders(<BulkEditUserModal {...defaultProps} />);

    expect(screen.getByTestId("user-edit-view")).toBeInTheDocument();
  });

  it("should show error when access token is missing", async () => {
    const user = userEvent.setup();
    renderWithProviders(<BulkEditUserModal {...defaultProps} accessToken={null} />);

    const submitButton = screen.getByRole("button", { name: "Submit" });
    await user.click(submitButton);

    await waitFor(() => {
      expect(NotificationsManager.fromBackend).toHaveBeenCalledWith("Access token not found");
    });
  });

  it("should call userBulkUpdateUserCall with correct payload for selected users", async () => {
    const user = userEvent.setup();
    renderWithProviders(<BulkEditUserModal {...defaultProps} />);

    const submitButton = screen.getByRole("button", { name: "Submit" });
    await user.click(submitButton);

    await waitFor(() => {
      expect(mockUserBulkUpdateUserCall).toHaveBeenCalledWith(
        "test-token",
        { user_role: "admin", max_budget: 100 },
        ["user1", "user2"],
      );
    });
  });

  it("should call userBulkUpdateUserCall with allUsers flag when update all users is enabled", async () => {
    const user = userEvent.setup();
    renderWithProviders(<BulkEditUserModal {...defaultProps} allowAllUsers={true} />);

    const updateAllCheckbox = screen.getByRole("checkbox", { name: /update all users/i });
    await user.click(updateAllCheckbox);

    const submitButton = screen.getByRole("button", { name: "Submit" });
    await user.click(submitButton);

    await waitFor(() => {
      expect(mockUserBulkUpdateUserCall).toHaveBeenCalledWith(
        "test-token",
        expect.objectContaining({ user_role: "admin", max_budget: 100 }),
        undefined,
        true,
      );
    });
  });


  it("should show success message after successful user update", async () => {
    const user = userEvent.setup();
    mockUserBulkUpdateUserCall.mockResolvedValue({
      results: [],
      total_requested: 2,
      successful_updates: 2,
      failed_updates: 0,
    });

    renderWithProviders(<BulkEditUserModal {...defaultProps} />);

    const submitButton = screen.getByRole("button", { name: "Submit" });
    await user.click(submitButton);

    await waitFor(() => {
      expect(NotificationsManager.success).toHaveBeenCalledWith("Updated 2 user(s)");
    });
  });

  it("should show success message for all users update", async () => {
    const user = userEvent.setup();
    mockUserBulkUpdateUserCall.mockResolvedValue({
      results: [],
      total_requested: 100,
      successful_updates: 100,
      failed_updates: 0,
    });

    renderWithProviders(<BulkEditUserModal {...defaultProps} allowAllUsers={true} />);

    const updateAllCheckbox = screen.getByRole("checkbox", { name: /update all users/i });
    await user.click(updateAllCheckbox);

    const submitButton = screen.getByRole("button", { name: "Submit" });
    await user.click(submitButton);

    await waitFor(() => {
      expect(NotificationsManager.success).toHaveBeenCalledWith("Updated all users (100 total)");
    });
  });


  it("should show error message when bulk update fails", async () => {
    const user = userEvent.setup();
    mockUserBulkUpdateUserCall.mockRejectedValueOnce(new Error("Update failed"));

    renderWithProviders(<BulkEditUserModal {...defaultProps} />);

    const submitButton = screen.getByRole("button", { name: "Submit" });
    await user.click(submitButton);

    await waitFor(() => {
      expect(NotificationsManager.fromBackend).toHaveBeenCalledWith("Failed to perform bulk operations");
    });
  });

  it("should call onSuccess and onCancel after successful update", async () => {
    const user = userEvent.setup();
    const onSuccess = vi.fn();
    const onCancel = vi.fn();

    renderWithProviders(<BulkEditUserModal {...defaultProps} onSuccess={onSuccess} onCancel={onCancel} />);

    const submitButton = screen.getByRole("button", { name: "Submit" });
    await user.click(submitButton);

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledTimes(1);
      expect(onCancel).toHaveBeenCalledTimes(1);
    });
  });

  it("should truncate long user IDs in table", () => {
    const longUserId = "a".repeat(30);
    const propsWithLongId = {
      ...defaultProps,
      selectedUsers: [{ user_id: longUserId, user_email: "test@example.com", user_role: "user", max_budget: null }],
    };

    renderWithProviders(<BulkEditUserModal {...propsWithLongId} />);

    expect(screen.getByText(new RegExp(`${longUserId.slice(0, 20)}...`))).toBeInTheDocument();
  });

  it("should display no email text when user email is missing", () => {
    const propsWithoutEmail = {
      ...defaultProps,
      selectedUsers: [{ user_id: "user1", user_email: null, user_role: "user", max_budget: null }],
    };

    renderWithProviders(<BulkEditUserModal {...propsWithoutEmail} />);

    expect(screen.getByText("No email")).toBeInTheDocument();
  });

  it("should display role label from possibleUIRoles when available", () => {
    renderWithProviders(<BulkEditUserModal {...defaultProps} />);

    expect(screen.getByText("Admin")).toBeInTheDocument();
    expect(screen.getByText("User")).toBeInTheDocument();
  });

  it("should display role key when ui_label is not available", () => {
    const propsWithoutUIRoles = {
      ...defaultProps,
      possibleUIRoles: null,
    };

    renderWithProviders(<BulkEditUserModal {...propsWithoutUIRoles} />);

    expect(screen.getByText("user")).toBeInTheDocument();
    expect(screen.getByText("admin")).toBeInTheDocument();
  });
});
