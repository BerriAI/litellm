import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import CreateKeyPage from "./page";

interface KeyRow {
  token: string;
}

const { mockReplace, mockUseKeys, mockMigratedHref, state } = vi.hoisted(() => {
  const state = {
    login: "success" as string | null,
    userRole: "Internal User",
    keys: [] as KeyRow[],
    returnUrl: null as string | null,
  };
  return {
    state,
    mockReplace: vi.fn(),
    mockMigratedHref: vi.fn((segment: string) => `/mocked-ui/${segment}`),
    mockUseKeys: vi.fn(() => ({
      data: { keys: state.keys, total_count: state.keys.length },
      isLoading: false,
    })),
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mockReplace }),
  useSearchParams: () => ({ get: (key: string) => (key === "login" ? state.login : null) }),
}));
vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({
    authLoading: false,
    token: "tok",
    userRole: state.userRole,
    userID: "user-1",
  }),
}));
vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({ useKeys: mockUseKeys }));
vi.mock("@/app/(dashboard)/api-keys/ApiKeysDashboard", () => ({
  default: () => <div data-testid="api-keys-dashboard" />,
}));
vi.mock("@/components/common_components/LoadingScreen", () => ({
  default: () => <div data-testid="loading-screen" />,
}));
vi.mock("@/components/networking", () => ({ proxyBaseUrl: "" }));
vi.mock("@/utils/migratedPages", () => ({ MIGRATED_PAGES: {}, migratedHref: mockMigratedHref }));
vi.mock("@/utils/returnUrlUtils", () => ({
  buildLoginUrlWithReturn: (u: string) => u,
  consumeReturnUrl: () => state.returnUrl,
  getLoginUrl: () => "/login",
  isValidReturnUrl: () => true,
  normalizeUrlForCompare: (u: string) => u,
  storeReturnUrl: () => undefined,
}));

const realLocation = window.location;
const mockLocationReplace = vi.fn();

describe("dashboard landing", () => {
  beforeEach(() => {
    Object.defineProperty(window, "location", {
      configurable: true,
      value: {
        origin: "http://localhost:3000",
        href: "http://localhost:3000/ui/?login=success",
        replace: mockLocationReplace,
      },
    });
  });

  afterEach(() => {
    Object.defineProperty(window, "location", { configurable: true, value: realLocation });
    state.login = "success";
    state.userRole = "Internal User";
    state.keys = [];
    state.returnUrl = null;
    mockReplace.mockClear();
    mockUseKeys.mockClear();
    mockMigratedHref.mockClear();
    mockLocationReplace.mockClear();
  });

  it.each(["Internal User", "Internal Viewer", "Admin", "Admin Viewer", "Org Admin", ""])(
    "lands a keyless %s on the keys dashboard, never on the MCP connect page",
    (role) => {
      state.userRole = role;
      render(<CreateKeyPage />);
      expect(screen.getByTestId("api-keys-dashboard")).toBeInTheDocument();
      expect(screen.queryByTestId("loading-screen")).not.toBeInTheDocument();
      expect(mockReplace).not.toHaveBeenCalled();
      expect(mockMigratedHref).not.toHaveBeenCalledWith("connect");
    },
  );

  it("lands a user who already owns a key on the keys dashboard", () => {
    state.keys = [{ token: "sk-abc" }];
    render(<CreateKeyPage />);
    expect(screen.getByTestId("api-keys-dashboard")).toBeInTheDocument();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("never looks a user's keys up to decide where the landing goes", () => {
    render(<CreateKeyPage />);
    expect(mockUseKeys).not.toHaveBeenCalled();
  });

  it("still sends the user to an explicit stored return URL", () => {
    state.returnUrl = "/ui/models-and-endpoints";
    render(<CreateKeyPage />);
    expect(mockLocationReplace).toHaveBeenCalledWith("http://localhost:3000/ui/models-and-endpoints");
  });
});
