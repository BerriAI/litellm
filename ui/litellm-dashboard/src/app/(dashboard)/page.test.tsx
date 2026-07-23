import { afterEach, describe, expect, it, vi } from "vitest";
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
    keysLoading: false,
    returnUrl: null as string | null,
  };
  return {
    state,
    mockReplace: vi.fn(),
    mockMigratedHref: vi.fn((segment: string) => `/mocked-ui/${segment}`),
    mockUseKeys: vi.fn((_page: number, _size: number, _opts: unknown, _enabled: boolean) => ({
      data: state.keysLoading ? undefined : { keys: state.keys, total_count: state.keys.length },
      isLoading: state.keysLoading,
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

describe("dashboard landing keyless redirect", () => {
  afterEach(() => {
    state.login = "success";
    state.userRole = "Internal User";
    state.keys = [];
    state.keysLoading = false;
    state.returnUrl = null;
    mockReplace.mockClear();
    mockUseKeys.mockClear();
    mockMigratedHref.mockClear();
  });

  it("sends a keyless non-admin to the connect page after login", () => {
    render(<CreateKeyPage />);
    expect(mockReplace).toHaveBeenCalledWith("/mocked-ui/connect");
    expect(screen.queryByTestId("api-keys-dashboard")).not.toBeInTheDocument();
  });

  it("leaves an admin with no keys on the dashboard", () => {
    state.userRole = "Admin";
    render(<CreateKeyPage />);
    expect(mockReplace).not.toHaveBeenCalled();
    expect(screen.getByTestId("api-keys-dashboard")).toBeInTheDocument();
  });

  it("leaves a user who already has a key on the dashboard", () => {
    state.keys = [{ token: "sk-abc" }];
    render(<CreateKeyPage />);
    expect(mockReplace).not.toHaveBeenCalled();
    expect(screen.getByTestId("api-keys-dashboard")).toBeInTheDocument();
  });

  it("does not redirect outside the post-login landing, and skips the key lookup entirely", () => {
    state.login = null;
    render(<CreateKeyPage />);
    expect(mockReplace).not.toHaveBeenCalled();
    expect(screen.getByTestId("api-keys-dashboard")).toBeInTheDocument();
    expect(mockUseKeys.mock.calls[0][3]).toBe(false);
  });

  it("holds the loading screen while the key lookup is in flight", () => {
    state.keysLoading = true;
    render(<CreateKeyPage />);
    expect(screen.getByTestId("loading-screen")).toBeInTheDocument();
    expect(screen.queryByTestId("api-keys-dashboard")).not.toBeInTheDocument();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("yields to an explicit return URL instead of the connect redirect", () => {
    state.returnUrl = "/ui/models-and-endpoints";
    render(<CreateKeyPage />);
    expect(mockReplace).not.toHaveBeenCalledWith("/mocked-ui/connect");
  });
});
