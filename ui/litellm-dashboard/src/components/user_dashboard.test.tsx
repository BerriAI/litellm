import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import { cleanup } from "@testing-library/react";
import React from "react";
import { renderWithProviders } from "../../tests/test-utils";

// Track addEventListener/removeEventListener calls for "beforeunload"
const addEventListenerSpy = vi.spyOn(window, "addEventListener");
const removeEventListenerSpy = vi.spyOn(window, "removeEventListener");

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
}));

// Mock networking with importOriginal so all exports are available
vi.mock("./networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./networking")>();
  return {
    ...actual,
    getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost:4000"),
    getProxyUISettings: vi.fn().mockResolvedValue({}),
    keyInfoCall: vi.fn().mockResolvedValue({}),
    modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
    userInfoCall: vi.fn().mockResolvedValue({ user_info: {}, keys: [], teams: [] }),
  };
});

// Mock jwt-decode to return a valid token structure
vi.mock("jwt-decode", () => ({
  jwtDecode: vi.fn().mockReturnValue({
    key: "test-access-token",
    user_role: "proxy_admin",
    user_email: "test@example.com",
    exp: Math.floor(Date.now() / 1000) + 3600,
  }),
}));

// Mock cookie utility
vi.mock("@/utils/cookieUtils", () => ({
  clearTokenCookies: vi.fn(),
}));

// Mock fetchTeams
vi.mock("./common_components/fetch_teams", () => ({
  fetchTeams: vi.fn(),
}));

// Mock heavy child components to isolate UserDashboard behavior
vi.mock("./organisms/create_key_button", () => ({
  default: () => <div data-testid="create-key-mock" />,
}));

vi.mock("./VirtualKeysPage/VirtualKeysTable", () => ({
  VirtualKeysTable: () => <div data-testid="virtual-keys-table-mock" />,
}));

vi.mock("../app/onboarding/page", () => ({
  default: () => <div data-testid="onboarding-mock" />,
}));

// Provide a token cookie so the component doesn't redirect to login
Object.defineProperty(document, "cookie", {
  writable: true,
  value: "token=fake-jwt-token",
});

import UserDashboard from "./user_dashboard";

const defaultProps = {
  userID: "user-1",
  userRole: "Admin",
  userEmail: "test@example.com",
  teams: [] as any[],
  keys: [] as any[],
  setUserRole: vi.fn(),
  setUserEmail: vi.fn(),
  setTeams: vi.fn(),
  setKeys: vi.fn(),
  premiumUser: false,
  organizations: [] as any[],
  addKey: vi.fn(),
  createClicked: false,
};

function renderDashboard(props = {}) {
  return renderWithProviders(<UserDashboard {...defaultProps} {...props} />);
}

describe("UserDashboard beforeunload listener", () => {
  beforeEach(() => {
    addEventListenerSpy.mockClear();
    removeEventListenerSpy.mockClear();
  });

  afterEach(() => {
    cleanup();
  });

  it("registers exactly one beforeunload listener on mount", () => {
    renderDashboard();

    const beforeUnloadCalls = addEventListenerSpy.mock.calls.filter(
      ([event]) => event === "beforeunload",
    );
    expect(beforeUnloadCalls).toHaveLength(1);
  });

  it("does not add duplicate listeners on re-render", () => {
    const { rerender } = renderWithProviders(<UserDashboard {...defaultProps} />);

    addEventListenerSpy.mockClear();

    // Re-render with different props to trigger a render cycle
    rerender(<UserDashboard {...defaultProps} userEmail="updated@example.com" />);

    const beforeUnloadCalls = addEventListenerSpy.mock.calls.filter(
      ([event]) => event === "beforeunload",
    );
    expect(beforeUnloadCalls).toHaveLength(0);
  });

  it("removes the beforeunload listener on unmount", () => {
    const { unmount } = renderDashboard();

    unmount();

    const removeCalls = removeEventListenerSpy.mock.calls.filter(
      ([event]) => event === "beforeunload",
    );
    expect(removeCalls).toHaveLength(1);
  });
});
