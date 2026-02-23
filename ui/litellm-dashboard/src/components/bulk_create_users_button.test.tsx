import { render } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import BulkCreateUsersButton from "./bulk_create_users_button";

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
  it("should render", () => {
    const { getByText } = render(<BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />);
    expect(getByText("+ Bulk Invite Users")).toBeInTheDocument();
  });
});
