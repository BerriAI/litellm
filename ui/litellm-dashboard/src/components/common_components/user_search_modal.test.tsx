import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, waitFor } from "../../../tests/test-utils";
import UserSearchModal from "./user_search_modal";
import * as networking from "@/components/networking";

// Mock the networking module
vi.mock("@/components/networking", () => ({
  userFilterUICall: vi.fn(),
}));

describe("UserSearchModal", () => {
  const defaultProps = {
    isVisible: true,
    onCancel: vi.fn(),
    onSubmit: vi.fn(),
    accessToken: "test-token",
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.userFilterUICall).mockResolvedValue([]);
  });

  it("renders the modal with default title", () => {
    const { getByText } = renderWithProviders(<UserSearchModal {...defaultProps} />);
    expect(getByText("Add Team Member")).toBeInTheDocument();
  });

  it("renders with custom title", () => {
    const { getByText } = renderWithProviders(<UserSearchModal {...defaultProps} title="Add Organization Member" />);
    expect(getByText("Add Organization Member")).toBeInTheDocument();
  });

  it("passes organization_id to userFilterUICall when searching and organizationId is provided", async () => {
    const user = userEvent.setup();
    const organizationId = "org-123";

    vi.mocked(networking.userFilterUICall).mockResolvedValue([
      { user_id: "user-1", user_email: "user1@example.com" },
    ]);

    renderWithProviders(<UserSearchModal {...defaultProps} organizationId={organizationId} />);

    // Get the email input by its id (Ant Design Select uses id for the input)
    const emailInput = document.getElementById("user_email") as HTMLInputElement;
    expect(emailInput).not.toBeNull();

    await user.click(emailInput);
    await user.type(emailInput, "user1");

    await waitFor(() => {
      expect(networking.userFilterUICall).toHaveBeenCalled();
    });

    // Verify the URLSearchParams include organization_id
    const callArgs = vi.mocked(networking.userFilterUICall).mock.calls[0];
    const params = callArgs[1] as URLSearchParams;
    expect(params.get("organization_id")).toBe(organizationId);
    expect(params.get("user_email")).toBe("user1");
  });

  it("does not pass organization_id when organizationId is not provided", async () => {
    const user = userEvent.setup();

    vi.mocked(networking.userFilterUICall).mockResolvedValue([
      { user_id: "user-1", user_email: "user1@example.com" },
    ]);

    renderWithProviders(<UserSearchModal {...defaultProps} />);

    // Get the email input by its id
    const emailInput = document.getElementById("user_email") as HTMLInputElement;
    expect(emailInput).not.toBeNull();

    await user.click(emailInput);
    await user.type(emailInput, "user1");

    await waitFor(() => {
      expect(networking.userFilterUICall).toHaveBeenCalled();
    });

    // Verify the URLSearchParams do NOT include organization_id
    const callArgs = vi.mocked(networking.userFilterUICall).mock.calls[0];
    const params = callArgs[1] as URLSearchParams;
    expect(params.get("organization_id")).toBeNull();
    expect(params.get("user_email")).toBe("user1");
  });

  it("does not pass organization_id when organizationId is null", async () => {
    const user = userEvent.setup();

    vi.mocked(networking.userFilterUICall).mockResolvedValue([
      { user_id: "user-1", user_email: "user1@example.com" },
    ]);

    renderWithProviders(<UserSearchModal {...defaultProps} organizationId={null} />);

    // Get the email input by its id
    const emailInput = document.getElementById("user_email") as HTMLInputElement;
    expect(emailInput).not.toBeNull();

    await user.click(emailInput);
    await user.type(emailInput, "user1");

    await waitFor(() => {
      expect(networking.userFilterUICall).toHaveBeenCalled();
    });

    // Verify the URLSearchParams do NOT include organization_id
    const callArgs = vi.mocked(networking.userFilterUICall).mock.calls[0];
    const params = callArgs[1] as URLSearchParams;
    expect(params.get("organization_id")).toBeNull();
  });

  it("passes organization_id when searching by user_id", async () => {
    const user = userEvent.setup();
    const organizationId = "org-456";

    vi.mocked(networking.userFilterUICall).mockResolvedValue([
      { user_id: "user-1", user_email: "user1@example.com" },
    ]);

    renderWithProviders(<UserSearchModal {...defaultProps} organizationId={organizationId} />);

    // Get the user_id input by its id
    const userIdInput = document.getElementById("user_id") as HTMLInputElement;
    expect(userIdInput).not.toBeNull();

    await user.click(userIdInput);
    await user.type(userIdInput, "user-1");

    await waitFor(() => {
      expect(networking.userFilterUICall).toHaveBeenCalled();
    });

    // Verify the URLSearchParams include organization_id
    const callArgs = vi.mocked(networking.userFilterUICall).mock.calls[0];
    const params = callArgs[1] as URLSearchParams;
    expect(params.get("organization_id")).toBe(organizationId);
    expect(params.get("user_id")).toBe("user-1");
  });
});
