/* @vitest-environment jsdom */
import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import useAuthorized from "./useAuthorized";

const { replaceMock, clearTokenCookiesMock, getProxyBaseUrlMock } = vi.hoisted(() => ({
  replaceMock: vi.fn(),
  clearTokenCookiesMock: vi.fn(),
  getProxyBaseUrlMock: vi.fn(() => "http://proxy.example"),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: replaceMock,
  }),
}));

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: getProxyBaseUrlMock,
}));

vi.mock("@/utils/cookieUtils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/utils/cookieUtils")>();
  return {
    ...actual,
    clearTokenCookies: clearTokenCookiesMock,
  };
});

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
    clearCookie();
  });

  it("should decode the token and expose user details", () => {
    const token = createJwt({
      key: "api-key-123",
      user_id: "user-1",
      user_email: "user@example.com",
      user_role: "app_admin",
      premium_user: true,
      disabled_non_admin_personal_key_creation: false,
      login_method: "username_password",
    });
    document.cookie = `token=${token}; path=/;`;

    const { result } = renderHook(() => useAuthorized());

    expect(result.current.token).toBe(token);
    expect(result.current.accessToken).toBe("api-key-123");
    expect(result.current.userId).toBe("user-1");
    expect(result.current.userEmail).toBe("user@example.com");
    expect(result.current.userRole).toBe("Admin");
    expect(result.current.premiumUser).toBe(true);
    expect(result.current.disabledPersonalKeyCreation).toBe(false);
    expect(result.current.showSSOBanner).toBe(true);
    expect(replaceMock).not.toHaveBeenCalled();
  });

  it("should clear cookies and redirect on an invalid token", () => {
    document.cookie = "token=invalid-token; path=/;";

    const { result } = renderHook(() => useAuthorized());

    expect(clearTokenCookiesMock).toHaveBeenCalled();
    expect(replaceMock).toHaveBeenCalledWith("http://proxy.example/ui/login");
    expect(result.current.accessToken).toBeNull();
    expect(result.current.userRole).toBe("Undefined Role");
  });
});
