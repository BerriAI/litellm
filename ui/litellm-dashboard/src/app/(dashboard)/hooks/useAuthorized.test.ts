/* @vitest-environment jsdom */
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider } from "@/contexts/AuthContext";
import useAuthorized from "./useAuthorized";

// Unmock useAuthorized to test the actual implementation
vi.unmock("@/app/(dashboard)/hooks/useAuthorized");

const { replaceMock, clearTokenCookiesMock, getProxyBaseUrlMock, getUiConfigMock, buildLoginUrlWithReturnMock } =
  vi.hoisted(() => ({
    replaceMock: vi.fn(),
    clearTokenCookiesMock: vi.fn(),
    getProxyBaseUrlMock: vi.fn(() => "http://proxy.example"),
    getUiConfigMock: vi.fn(),
    buildLoginUrlWithReturnMock: vi.fn((baseUrl: string) => baseUrl),
  }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: replaceMock,
  }),
}));

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    getProxyBaseUrl: getProxyBaseUrlMock,
    getUiConfig: getUiConfigMock,
  };
});

vi.mock("@/utils/cookieUtils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/utils/cookieUtils")>();
  return {
    ...actual,
    clearTokenCookies: clearTokenCookiesMock,
  };
});

vi.mock("@/utils/returnUrlUtils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/utils/returnUrlUtils")>();
  return {
    ...actual,
    buildLoginUrlWithReturn: buildLoginUrlWithReturnMock,
    storeReturnUrl: vi.fn(),
  };
});

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = createQueryClient();
  return React.createElement(
    QueryClientProvider,
    { client: queryClient },
    React.createElement(AuthProvider, null, children),
  );
};

const createJwt = (payload: Record<string, unknown>) => {
  const base64Url = btoa(JSON.stringify(payload)).replace(/=+$/, "").replace(/\+/g, "-").replace(/\//g, "_");
  return `eyJhbGciOiJub25lIn0.${base64Url}.signature`;
};

const clearCookie = () => {
  document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
};

const uiConfig = (overrides: Record<string, unknown> = {}) => ({
  server_root_path: "/",
  proxy_base_url: null,
  auto_redirect_to_sso: false,
  admin_ui_disabled: false,
  sso_configured: false,
  ...overrides,
});

const decodedPayload = {
  key: "api-key-123",
  user_id: "user-1",
  user_email: "user@example.com",
  user_role: "app_admin",
  premium_user: true,
  disabled_non_admin_personal_key_creation: false,
  login_method: "username_password",
};

describe("useAuthorized", () => {
  afterEach(() => {
    replaceMock.mockReset();
    clearTokenCookiesMock.mockReset();
    getProxyBaseUrlMock.mockClear();
    getUiConfigMock.mockReset();
    buildLoginUrlWithReturnMock.mockClear();
    clearCookie();
  });

  it("should decode the token and expose user details", async () => {
    getUiConfigMock.mockResolvedValue(uiConfig());

    const token = createJwt(decodedPayload);
    document.cookie = `token=${token}; path=/;`;

    const { result } = renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(result.current.token).toBe(token);
    });

    expect(result.current.isAuthorized).toBe(true);
    expect(result.current.accessToken).toBe("api-key-123");
    expect(result.current.userId).toBe("user-1");
    expect(result.current.userEmail).toBe("user@example.com");
    expect(result.current.userRole).toBe("Admin");
    expect(result.current.premiumUser).toBe(true);
    expect(result.current.disabledPersonalKeyCreation).toBe(false);
    expect(result.current.showSSOBanner).toBe(true);
    expect(replaceMock).not.toHaveBeenCalled();
    expect(clearTokenCookiesMock).not.toHaveBeenCalled();
  });

  it("should clear cookies and redirect on an undecodable token", async () => {
    getUiConfigMock.mockResolvedValue(uiConfig());

    document.cookie = "token=invalid-token; path=/;";

    const { result } = renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(replaceMock).toHaveBeenCalledWith("http://proxy.example/ui/login");
    });

    expect(clearTokenCookiesMock).toHaveBeenCalled();
    expect(result.current.token).toBeNull();
    expect(result.current.accessToken).toBeNull();
    expect(result.current.userRole).toBe("Undefined Role");
  });

  it("should redirect even with valid token if admin_ui_disabled is true", async () => {
    getUiConfigMock.mockResolvedValue(uiConfig({ admin_ui_disabled: true }));

    const token = createJwt(decodedPayload);
    document.cookie = `token=${token}; path=/;`;

    const { result } = renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(replaceMock).toHaveBeenCalledWith("http://proxy.example/ui/login");
    });

    expect(clearTokenCookiesMock).toHaveBeenCalled();
    expect(result.current.isAuthorized).toBe(false);
    expect(result.current.token).toBeNull();
    expect(result.current.accessToken).toBe("api-key-123");
    expect(result.current.userId).toBe("user-1");
    expect(result.current.userEmail).toBe("user@example.com");
  });

  it("should redirect when token is missing", async () => {
    getUiConfigMock.mockResolvedValue(uiConfig());

    // No token cookie set
    const { result } = renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(replaceMock).toHaveBeenCalledWith("http://proxy.example/ui/login");
    });

    expect(clearTokenCookiesMock).not.toHaveBeenCalled();
    expect(result.current.token).toBeNull();
  });

  it("should clear cookies and redirect when token is expired", async () => {
    getUiConfigMock.mockResolvedValue(uiConfig());

    const token = createJwt({ ...decodedPayload, exp: Math.floor(Date.now() / 1000) - 60 });
    document.cookie = `token=${token}; path=/;`;

    const { result } = renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(clearTokenCookiesMock).toHaveBeenCalled();
    });

    expect(replaceMock).toHaveBeenCalledWith("http://proxy.example/ui/login");
    expect(result.current.token).toBeNull();
    expect(result.current.accessToken).toBeNull();
  });
});
