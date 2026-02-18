/* @vitest-environment jsdom */
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import useAuthorized from "./useAuthorized";

// Unmock useAuthorized to test the actual implementation
vi.unmock("@/app/(dashboard)/hooks/useAuthorized");

const { replaceMock, clearTokenCookiesMock, getProxyBaseUrlMock, getUiConfigMock, decodeTokenMock, checkTokenValidityMock } = vi.hoisted(() => ({
  replaceMock: vi.fn(),
  clearTokenCookiesMock: vi.fn(),
  getProxyBaseUrlMock: vi.fn(() => "http://proxy.example"),
  getUiConfigMock: vi.fn(),
  decodeTokenMock: vi.fn(),
  checkTokenValidityMock: vi.fn(),
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

vi.mock("@/utils/jwtUtils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/utils/jwtUtils")>();
  return {
    ...actual,
    decodeToken: decodeTokenMock,
    checkTokenValidity: checkTokenValidityMock,
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
  return React.createElement(QueryClientProvider, { client: queryClient }, children);
};

const createJwt = (payload: Record<string, unknown>) => {
  const base64Url = btoa(JSON.stringify(payload)).replace(/=+$/, "").replace(/\+/g, "-").replace(/\//g, "_");
  return `eyJhbGciOiJub25lIn0.${base64Url}.signature`;
};

const clearCookie = () => {
  document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
};

describe("useAuthorized", () => {
  afterEach(() => {
    replaceMock.mockReset();
    clearTokenCookiesMock.mockReset();
    getProxyBaseUrlMock.mockClear();
    getUiConfigMock.mockReset();
    decodeTokenMock.mockReset();
    checkTokenValidityMock.mockReset();
    clearCookie();
  });

  it("should decode the token and expose user details", async () => {
    getUiConfigMock.mockResolvedValue({
      server_root_path: "/",
      proxy_base_url: null,
      auto_redirect_to_sso: false,
      admin_ui_disabled: false,
      sso_configured: false,
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
    
    decodeTokenMock.mockReturnValue(decodedPayload);
    checkTokenValidityMock.mockReturnValue(true);

    const token = createJwt(decodedPayload);
    document.cookie = `token=${token}; path=/;`;

    const { result } = renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(result.current.token).toBe(token);
    });

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

  it("should clear cookies and redirect on an invalid token", async () => {
    getUiConfigMock.mockResolvedValue({
      server_root_path: "/",
      proxy_base_url: null,
      auto_redirect_to_sso: false,
      admin_ui_disabled: false,
      sso_configured: false,
    });

    decodeTokenMock.mockReturnValue(null);
    checkTokenValidityMock.mockReturnValue(false);

    document.cookie = "token=invalid-token; path=/;";

    const { result } = renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(clearTokenCookiesMock).toHaveBeenCalled();
    });

    expect(replaceMock).toHaveBeenCalledWith("http://proxy.example/ui/login");
    expect(result.current.accessToken).toBeNull();
    expect(result.current.userRole).toBe("Undefined Role");
  });

  it("should redirect even with valid token if admin_ui_disabled is true", async () => {
    getUiConfigMock.mockResolvedValue({
      server_root_path: "/",
      proxy_base_url: null,
      auto_redirect_to_sso: false,
      admin_ui_disabled: true,
      sso_configured: false,
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

    decodeTokenMock.mockReturnValue(decodedPayload);
    checkTokenValidityMock.mockReturnValue(true);

    const token = createJwt(decodedPayload);
    document.cookie = `token=${token}; path=/;`;

    const { result } = renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(replaceMock).toHaveBeenCalledWith("http://proxy.example/ui/login");
    });

    expect(result.current.accessToken).toBe("api-key-123");
    expect(result.current.userId).toBe("user-1");
    expect(result.current.userEmail).toBe("user@example.com");
  });

  it("should redirect when token is missing", async () => {
    getUiConfigMock.mockResolvedValue({
      server_root_path: "/",
      proxy_base_url: null,
      auto_redirect_to_sso: false,
      admin_ui_disabled: false,
      sso_configured: false,
    });

    decodeTokenMock.mockReturnValue(null);
    checkTokenValidityMock.mockReturnValue(false);

    // No token cookie set
    const { result } = renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(replaceMock).toHaveBeenCalledWith("http://proxy.example/ui/login");
    });

    expect(clearTokenCookiesMock).not.toHaveBeenCalled();
    expect(result.current.token).toBeNull();
  });

  it("should clear cookies and redirect when token is expired", async () => {
    getUiConfigMock.mockResolvedValue({
      server_root_path: "/",
      proxy_base_url: null,
      auto_redirect_to_sso: false,
      admin_ui_disabled: false,
      sso_configured: false,
    });

    const decodedPayload = {
      key: "api-key-123",
      user_id: "user-1",
      user_email: "user@example.com",
      user_role: "app_admin",
    };

    decodeTokenMock.mockReturnValue(decodedPayload);
    checkTokenValidityMock.mockReturnValue(false);

    const token = createJwt(decodedPayload);
    document.cookie = `token=${token}; path=/;`;

    const { result } = renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(clearTokenCookiesMock).toHaveBeenCalled();
    });

    expect(replaceMock).toHaveBeenCalledWith("http://proxy.example/ui/login");
    expect(checkTokenValidityMock).toHaveBeenCalledWith(token);
  });
});
