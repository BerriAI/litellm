import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, it, expect, vi } from "vitest";
import BulkCreateUsersButton from "./bulk_create_users_button";
import * as networking from "./networking";

vi.mock("./networking", () => ({
  userCreateCall: vi.fn(),
  invitationCreateCall: vi.fn(),
  getProxyUISettings: vi.fn().mockResolvedValue({
    PROXY_BASE_URL: null,
    PROXY_LOGOUT_URL: null,
    DEFAULT_TEAM_DISABLED: false,
    SSO_ENABLED: false,
  }),
}));

vi.mock("./molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

describe("BulkCreateUsersButton", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    const { getByText } = render(<BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />);
    expect(getByText("+ Bulk Invite Users")).toBeInTheDocument();
  });

  it("should request virtual key creation for bulk invited users", async () => {
    const userCreateCall = vi.mocked(networking.userCreateCall);
    const invitationCreateCall = vi.mocked(networking.invitationCreateCall);

    userCreateCall.mockResolvedValue({ user_id: "user-1", key: "sk-test-key" });
    invitationCreateCall.mockResolvedValue({ id: "invite-1" });

    render(<BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />);

    fireEvent.click(screen.getByText("+ Bulk Invite Users"));

    const file = new File(["user_email,user_role\nuser@example.com,internal_user\n"], "users.csv", {
      type: "text/csv",
    });
    const fileInput = document.body.querySelector("input[type='file']");
    expect(fileInput).not.toBeNull();
    fireEvent.change(fileInput!, { target: { files: [file] } });

    const createButtons = await screen.findAllByRole("button", { name: "Create 1 Users" });
    fireEvent.click(createButtons[0]);

    await waitFor(() => {
      expect(userCreateCall).toHaveBeenCalledWith(
        "test-token",
        null,
        expect.objectContaining({
          user_email: "user@example.com",
          user_role: "internal_user",
          auto_create_key: true,
          models: ["no-default-models"],
        }),
      );
    });
  });

  it("should restrict non-admin users when models csv value parses to empty", async () => {
    const userCreateCall = vi.mocked(networking.userCreateCall);
    const invitationCreateCall = vi.mocked(networking.invitationCreateCall);

    userCreateCall.mockResolvedValue({ user_id: "user-1", key: "sk-test-key" });
    invitationCreateCall.mockResolvedValue({ id: "invite-1" });

    render(<BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />);

    fireEvent.click(screen.getByText("+ Bulk Invite Users"));

    const file = new File(['user_email,user_role,models\nuser@example.com,internal_user,", ,"\n'], "users.csv", {
      type: "text/csv",
    });
    const fileInput = document.body.querySelector("input[type='file']");
    expect(fileInput).not.toBeNull();
    fireEvent.change(fileInput!, { target: { files: [file] } });

    const createButtons = await screen.findAllByRole("button", { name: "Create 1 Users" });
    fireEvent.click(createButtons[0]);

    await waitFor(() => {
      expect(userCreateCall).toHaveBeenCalledWith(
        "test-token",
        null,
        expect.objectContaining({
          user_email: "user@example.com",
          user_role: "internal_user",
          auto_create_key: true,
          models: ["no-default-models"],
        }),
      );
    });
  });

  it("should fail the row when user creation returns no virtual key", async () => {
    const userCreateCall = vi.mocked(networking.userCreateCall);
    const invitationCreateCall = vi.mocked(networking.invitationCreateCall);

    userCreateCall.mockResolvedValue({ user_id: "user-1" });

    render(<BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />);

    fireEvent.click(screen.getByText("+ Bulk Invite Users"));

    const file = new File(["user_email,user_role\nuser@example.com,internal_user\n"], "users.csv", {
      type: "text/csv",
    });
    const fileInput = document.body.querySelector("input[type='file']");
    expect(fileInput).not.toBeNull();
    fireEvent.change(fileInput!, { target: { files: [file] } });

    const createButtons = await screen.findAllByRole("button", { name: "Create 1 Users" });
    fireEvent.click(createButtons[0]);

    await screen.findByText(JSON.stringify("User created but no virtual key was returned"));
    expect(invitationCreateCall).not.toHaveBeenCalled();
  });
});
