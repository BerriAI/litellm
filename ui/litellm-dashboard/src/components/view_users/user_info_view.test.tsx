import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import UserInfoView from "./user_info_view";

vi.mock("../networking", () => {
  const MOCK_USER_DATA = {
    user_id: "user-123",
    user_info: {
      user_email: "test@example.com",
      user_alias: "Test Alias",
      user_role: "admin",
      teams: [],
      models: [],
      max_budget: 100,
      budget_duration: "30d",
      spend: 0,
      metadata: {},
      created_at: "2025-01-01T00:00:00.000Z",
      updated_at: "2025-01-02T00:00:00.000Z",
    },
    keys: [],
    teams: [],
  };

  return {
    userInfoCall: vi.fn().mockResolvedValue(MOCK_USER_DATA),
    userDeleteCall: vi.fn(),
    userUpdateUserCall: vi.fn(),
    modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
    invitationCreateCall: vi.fn(),
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

  it("should render the loading state and then the user email", async () => {
    const { getByText, findAllByText } = render(<UserInfoView {...defaultProps} />);

    expect(getByText("Loading user data...")).toBeInTheDocument();

    const emails = await findAllByText("test@example.com");
    expect(emails.length).toBeGreaterThan(0);
  });

  it("should render the user alias", async () => {
    const { findAllByText } = render(<UserInfoView {...defaultProps} />);

    const aliases = await findAllByText("Test Alias");
    expect(aliases.length).toBeGreaterThan(0);
  });
});
