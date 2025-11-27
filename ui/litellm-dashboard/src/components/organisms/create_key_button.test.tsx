import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import CreateKey from "./create_key_button";

const mockKeyCreateCall = vi.fn().mockResolvedValue({
  key: "test-api-key",
  soft_budget: null,
});

vi.mock("./networking", () => ({
  keyCreateCall: mockKeyCreateCall,
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
  getPromptsList: vi.fn().mockResolvedValue({ prompts: [] }),
  proxyBaseUrl: "http://localhost:4000",
  getPossibleUserRoles: vi.fn().mockResolvedValue({
    Admin: { ui_label: "Admin" },
    User: { ui_label: "User" },
  }),
  userFilterUICall: vi.fn().mockResolvedValue([]),
  keyCreateServiceAccountCall: vi.fn().mockResolvedValue({
    key: "test-service-account-key",
    soft_budget: null,
  }),
  fetchMCPAccessGroups: vi.fn().mockResolvedValue([]),
}));

vi.mock("./molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    clear: vi.fn(),
  },
}));

describe("CreateKey", () => {
  const defaultProps = {
    userID: "test-user-id",
    team: null,
    userRole: "Admin",
    accessToken: "test-token",
    data: [],
    teams: [],
    addKey: vi.fn(),
    premiumUser: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mockKeyCreateCall.mockResolvedValue({
      key: "test-api-key",
      soft_budget: null,
    });
  });

  it("should render the CreateKey component", () => {
    render(<CreateKey {...defaultProps} />);
    expect(screen.getByRole("button", { name: /create new key/i })).toBeInTheDocument();
  });
});
