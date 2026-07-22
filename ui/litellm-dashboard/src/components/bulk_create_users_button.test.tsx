import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, it, expect, vi } from "vitest";
import Papa from "papaparse";
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

  const uploadCsv = (csv: string) => {
    const file = new File([csv], "users.csv", {
      type: "text/csv",
    });
    const dialog = screen.getByRole("dialog", { name: "Bulk Invite Users" });
    const fileInput = dialog.querySelector("input[type='file']");
    expect(fileInput).not.toBeNull();
    fireEvent.change(fileInput!, { target: { files: [file] } });
  };

  it("should render", () => {
    const { getByText } = render(<BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />);
    expect(getByText("+ Bulk Invite Users")).toBeInTheDocument();
  });

  it("should request virtual key creation and export returned keys for bulk invited users", async () => {
    const userCreateCall = vi.mocked(networking.userCreateCall);
    const invitationCreateCall = vi.mocked(networking.invitationCreateCall);
    const unparseSpy = vi.spyOn(Papa, "unparse");
    Object.defineProperty(window.URL, "createObjectURL", {
      writable: true,
      value: vi.fn(() => "blob:test-results"),
    });
    Object.defineProperty(window.URL, "revokeObjectURL", {
      writable: true,
      value: vi.fn(),
    });

    userCreateCall.mockResolvedValue({ user_id: "user-1", key: "sk-test-key" });
    invitationCreateCall.mockResolvedValue({ id: "invite-1" });

    render(<BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />);

    fireEvent.click(screen.getByText("+ Bulk Invite Users"));

    uploadCsv("user_email,user_role\n@evil.example.com,internal_user\n");

    const createButtons = await screen.findAllByRole("button", { name: "Create 1 Users" });
    fireEvent.click(createButtons[0]);

    await waitFor(() => {
      expect(userCreateCall).toHaveBeenCalledWith(
        "test-token",
        null,
        expect.objectContaining({
          user_email: "@evil.example.com",
          user_role: "internal_user",
          auto_create_key: true,
          models: ["no-default-models"],
        }),
      );
    });

    await screen.findByText("User creation complete");
    fireEvent.click(screen.getByRole("button", { name: /Download User Credentials/i }));

    const lastUnparseCall = unparseSpy.mock.calls[unparseSpy.mock.calls.length - 1];
    expect(lastUnparseCall[0]).toEqual([
      expect.objectContaining({
        user_email: "@evil.example.com",
        status: "success",
        key: "sk-test-key",
      }),
    ]);
    expect(lastUnparseCall[1]).toEqual({ escapeFormulae: true });
    expect(lastUnparseCall[0]).not.toEqual([
      expect.objectContaining({
        key: "user-1",
      }),
    ]);
  });

  it("should restrict non-admin users when models csv value parses to empty", async () => {
    const userCreateCall = vi.mocked(networking.userCreateCall);
    const invitationCreateCall = vi.mocked(networking.invitationCreateCall);

    userCreateCall.mockResolvedValue({ user_id: "user-1", key: "sk-test-key" });
    invitationCreateCall.mockResolvedValue({ id: "invite-1" });

    render(<BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />);

    fireEvent.click(screen.getByText("+ Bulk Invite Users"));

    uploadCsv('user_email,user_role,models\nuser@example.com,internal_user,", ,"\n');

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

    uploadCsv("user_email,user_role\nuser@example.com,internal_user\n");

    const createButtons = await screen.findAllByRole("button", { name: "Create 1 Users" });
    fireEvent.click(createButtons[0]);

    await screen.findByText(JSON.stringify("User created but no virtual key was returned"));
    expect(invitationCreateCall).not.toHaveBeenCalled();
  });
});
