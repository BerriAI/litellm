import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import UserInfoView from "./user_info_view";

vi.mock("../networking", () => {
  const MOCK_USER_DATA = {
    user_id: "user-123",
    user_email: "test@example.com",
    user_alias: "Test Alias",
    user_role: "admin",
    spend: 0,
    max_budget: 100,
    models: [],
    budget_duration: "30d",
    budget_reset_at: null,
    metadata: {},
    created_at: "2025-01-01T00:00:00.000Z",
    updated_at: "2025-01-02T00:00:00.000Z",
    sso_user_id: null,
    teams: [],
  };

  return {
    userGetInfoV2: vi.fn().mockResolvedValue(MOCK_USER_DATA),
    userDeleteCall: vi.fn(),
    userUpdateUserCall: vi.fn(),
    modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
    invitationCreateCall: vi.fn(),
    teamInfoCall: vi.fn().mockResolvedValue({ team_alias: "Test Team" }),
    getProxyBaseUrl: () => "https://litellm.test",
  };
});

describe("UserInfoView", () => {
  const defaultProps = {
    userId: "user-123",
    onClose: vi.fn(),
    accessToken: "test-token",
    userRole: null,
    possibleUIRoles: null,
  };

  it("should render the loading state", () => {
    render(<UserInfoView {...defaultProps} />);

    expect(screen.getByText("Loading user data...")).toBeInTheDocument();
  });

  it("should render the user email after loading", async () => {
    render(<UserInfoView {...defaultProps} />);

    const emails = await screen.findAllByText("test@example.com");
    expect(emails.length).toBeGreaterThan(0);
    expect(screen.queryByText("Loading user data...")).not.toBeInTheDocument();
  });

  it("should render the user alias after loading", async () => {
    render(<UserInfoView {...defaultProps} />);

    const aliases = await screen.findAllByText("Test Alias");
    expect(aliases.length).toBeGreaterThan(0);
  });
});
