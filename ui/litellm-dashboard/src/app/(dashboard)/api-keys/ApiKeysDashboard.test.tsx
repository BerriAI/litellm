import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { teamListCall, authorizedSession } = vi.hoisted(() => ({
  teamListCall: vi.fn(() => new Promise(() => {})),
  authorizedSession: vi.fn(),
}));

const session = (overrides: { userRole?: string; isViewOnly?: boolean } = {}) => ({
  isLoading: false,
  isAuthorized: true,
  token: "jwt",
  accessToken: "sk-access",
  userId: "u-123",
  userEmail: "admin@example.com",
  userRole: "Admin",
  isViewOnly: false,
  premiumUser: false,
  disabledPersonalKeyCreation: false,
  showSSOBanner: false,
  ...overrides,
});

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => authorizedSession(),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  teamListCall,
}));

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(""),
}));

vi.mock("@/components/VirtualKeysPage/VirtualKeysTable", () => ({
  VirtualKeysTable: ({ headerActions }: { headerActions?: React.ReactNode }) => (
    <div>
      {headerActions}
      <table aria-label="Virtual Keys" />
    </div>
  ),
}));

vi.mock("@/components/organisms/create_key_button", () => ({
  default: () => <button type="button">Create Key</button>,
}));

import ApiKeysDashboard from "./ApiKeysDashboard";

describe("ApiKeysDashboard", () => {
  beforeEach(() => {
    teamListCall.mockClear();
    authorizedSession.mockReturnValue(session());
    sessionStorage.clear();
  });

  it("scopes the team list to the signed-in user for non-admin roles", () => {
    authorizedSession.mockReturnValue(session({ userRole: "Internal User" }));
    render(<ApiKeysDashboard />);

    expect(teamListCall).toHaveBeenCalledWith("sk-access", 1, 100, { userID: "u-123" });
  });

  it("renders the keys table with a Create Key action for roles that can write", () => {
    render(<ApiKeysDashboard />);

    expect(screen.getByRole("table", { name: "Virtual Keys" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create Key" })).toBeInTheDocument();
  });

  it("hides Create Key for view-only roles", () => {
    authorizedSession.mockReturnValue(session({ isViewOnly: true }));
    render(<ApiKeysDashboard />);

    expect(screen.getByRole("table", { name: "Virtual Keys" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create Key" })).not.toBeInTheDocument();
  });

  it("leaves other pages' session state intact when the tab reloads", () => {
    sessionStorage.setItem("chatHistory", '[{"role":"user","content":"hi"}]');
    sessionStorage.setItem("selectedModel", "gpt-5.5");
    render(<ApiKeysDashboard />);

    window.dispatchEvent(new Event("beforeunload"));

    expect(sessionStorage.getItem("chatHistory")).toBe('[{"role":"user","content":"hi"}]');
    expect(sessionStorage.getItem("selectedModel")).toBe("gpt-5.5");
  });
});
