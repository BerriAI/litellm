/* @vitest-environment jsdom */
import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import useAuthorized from "./useAuthorized";
import { useUIConfig } from "./uiConfig/useUIConfig";
import { useCurrentUser } from "./currentUser/useCurrentUser";

// Unmock useAuthorized to test the actual implementation
vi.unmock("@/app/(dashboard)/hooks/useAuthorized");

const {
  replaceMock,
  clearTokenCookiesMock,
  getProxyBaseUrlMock,
  getUiConfigMock,
  decodeTokenMock,
  checkTokenValidityMock,
  buildLoginUrlWithReturnMock,
} = vi.hoisted(() => ({
  replaceMock: vi.fn(),
  clearTokenCookiesMock: vi.fn(),
  getProxyBaseUrlMock: vi.fn(() => "http://proxy.example"),
  getUiConfigMock: vi.fn(),
  decodeTokenMock: vi.fn(),
  checkTokenValidityMock: vi.fn(),
  buildLoginUrlWithReturnMock: vi.fn((baseUrl: string) => baseUrl),
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

vi.mock("@/utils/returnUrlUtils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/utils/returnUrlUtils")>();
  return {
    ...actual,
    buildLoginUrlWithReturn: buildLoginUrlWithReturnMock,
    storeReturnUrl: vi.fn(),
  };
});
const fetchMock = vi.fn<(request: Request) => Promise<Response>>();

const CURRENT_USER_PATH = "/management/v1/users/me";

const currentUserRequests = () =>
  fetchMock.mock.calls
    .map(([request]) => request)
    .filter((request) => new URL(request.url).pathname === CURRENT_USER_PATH);

type CurrentUserBody = { user_id: string; user_email: string | null; user_role: string | null };

const serveCurrentUser = (reply: CurrentUserBody | ((request: Request) => CurrentUserBody)) => {
  fetchMock.mockImplementation(async (request) => Response.json(typeof reply === "function" ? reply(request) : reply));
  vi.stubGlobal("fetch", fetchMock);
};

const stallCurrentUser = () => {
  fetchMock.mockImplementation(() => new Promise<Response>(() => {}));
  vi.stubGlobal("fetch", fetchMock);
};

const failCurrentUser = (status: number) => {
  fetchMock.mockImplementation(async () => Response.json({ detail: "rejected" }, { status }));
  vi.stubGlobal("fetch", fetchMock);
};

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
  const originalLocation = window.location;

  beforeEach(() => {
    Object.defineProperty(window, "location", {
      value: {
        href: "http://proxy.example/ui/?page=api-keys",
        origin: "http://proxy.example",
        hostname: "proxy.example",
        pathname: "/ui/",
        search: "?page=api-keys",
        protocol: "http:",
        replace: replaceMock,
      },
      writable: true,
    });
  });

  afterEach(() => {
    Object.defineProperty(window, "location", {
      value: originalLocation,
      writable: true,
    });
    replaceMock.mockReset();
    clearTokenCookiesMock.mockReset();
    getProxyBaseUrlMock.mockClear();
    getUiConfigMock.mockReset();
    decodeTokenMock.mockReset();
    checkTokenValidityMock.mockReset();
    buildLoginUrlWithReturnMock.mockClear();
    vi.unstubAllGlobals();
    fetchMock.mockReset();
    clearCookie();
  });

  it("should take user details from the server and the rest from the token", async () => {
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
      user_role: "proxy_admin",
      premium_user: true,
      disabled_non_admin_personal_key_creation: false,
      login_method: "username_password",
    };

    decodeTokenMock.mockReturnValue(decodedPayload);
    checkTokenValidityMock.mockReturnValue(true);
    serveCurrentUser({ user_id: "user-1", user_email: "user@example.com", user_role: "proxy_admin" });

    const token = createJwt(decodedPayload);
    document.cookie = `token=${token}; path=/;`;

    const { result } = renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(result.current.userId).toBe("user-1");
    });

    expect(currentUserRequests().map((request) => request.headers.get("Authorization"))).toEqual([
      "Bearer api-key-123",
    ]);
    expect(result.current.token).toBe(token);
    expect(result.current.accessToken).toBe("api-key-123");
    expect(result.current.userId).toBe("user-1");
    expect(result.current.userEmail).toBe("user@example.com");
    expect(result.current.userRole).toBe("Admin");
    expect(result.current.userRoleLabel).toBe("Admin");
    expect(result.current.isViewOnly).toBe(false);
    expect(result.current.premiumUser).toBe(true);
    expect(result.current.disabledPersonalKeyCreation).toBe(false);
    expect(result.current.showSSOBanner).toBe(true);
    expect(replaceMock).not.toHaveBeenCalled();
    expect(clearTokenCookiesMock).not.toHaveBeenCalled();
  });

  it("sends the first lookup on the auth header named in the session cookie", async () => {
    const uiConfig = {
      server_root_path: "/",
      proxy_base_url: null,
      auto_redirect_to_sso: false,
      admin_ui_disabled: false,
      sso_configured: false,
    };
    getUiConfigMock.mockResolvedValue(uiConfig);

    const decodedPayload = {
      key: "api-key-custom-header",
      user_id: "user-3",
      user_role: "internal_user",
      auth_header_name: "X-Litellm-Key",
    };

    decodeTokenMock.mockReturnValue(decodedPayload);
    checkTokenValidityMock.mockReturnValue(true);
    serveCurrentUser({ user_id: "user-3", user_email: null, user_role: "internal_user" });
    document.cookie = `token=${createJwt(decodedPayload)}; path=/;`;

    const { result } = renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(result.current.userId).toBe("user-3");
    });

    expect(currentUserRequests().map((request) => request.headers.get("X-Litellm-Key"))).toEqual([
      "Bearer api-key-custom-header",
    ]);
  });

  it("should present proxy_admin_viewer as Admin while flagging it view-only", async () => {
    getUiConfigMock.mockResolvedValue({
      server_root_path: "/",
      proxy_base_url: null,
      auto_redirect_to_sso: false,
      admin_ui_disabled: false,
      sso_configured: false,
    });

    const decodedPayload = {
      key: "api-key-456",
      user_id: "user-2",
      user_email: "viewer@example.com",
      user_role: "proxy_admin_viewer",
      premium_user: true,
      disabled_non_admin_personal_key_creation: false,
      login_method: "username_password",
    };

    decodeTokenMock.mockReturnValue(decodedPayload);
    checkTokenValidityMock.mockReturnValue(true);
    serveCurrentUser({
      user_id: "user-2",
      user_email: "viewer@example.com",
      user_role: "proxy_admin_viewer",
    });

    const token = createJwt(decodedPayload);
    document.cookie = `token=${token}; path=/;`;

    const { result } = renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(result.current.userId).toBe("user-2");
    });

    expect(result.current.userRole).toBe("Admin");
    expect(result.current.userRoleLabel).toBe("Admin Viewer");
    expect(result.current.isViewOnly).toBe(true);
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

    expect(replaceMock).toHaveBeenCalledWith("http://proxy.example/ui/login/");
    expect(result.current.accessToken).toBeNull();
    expect(result.current.userRole).toBe("Undefined Role");
    expect(currentUserRequests()).toHaveLength(0);
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
      user_role: "proxy_admin",
      premium_user: true,
      disabled_non_admin_personal_key_creation: false,
      login_method: "username_password",
    };

    decodeTokenMock.mockReturnValue(decodedPayload);
    checkTokenValidityMock.mockReturnValue(true);
    serveCurrentUser({ user_id: "user-1", user_email: "user@example.com", user_role: "proxy_admin" });

    const token = createJwt(decodedPayload);
    document.cookie = `token=${token}; path=/;`;

    const { result } = renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(replaceMock).toHaveBeenCalledWith("http://proxy.example/ui/login/");
    });

    expect(result.current.accessToken).toBe("api-key-123");
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
      expect(replaceMock).toHaveBeenCalledWith("http://proxy.example/ui/login/");
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
      user_role: "proxy_admin",
    };

    decodeTokenMock.mockReturnValue(decodedPayload);
    checkTokenValidityMock.mockReturnValue(false);

    const token = createJwt(decodedPayload);
    document.cookie = `token=${token}; path=/;`;

    renderHook(() => useAuthorized(), { wrapper });

    await waitFor(() => {
      expect(clearTokenCookiesMock).toHaveBeenCalled();
    });

    expect(replaceMock).toHaveBeenCalledWith("http://proxy.example/ui/login/");
    expect(checkTokenValidityMock).toHaveBeenCalledWith(token);
    expect(currentUserRequests()).toHaveLength(0);
  });

  describe("identity from the server", () => {
    const uiConfig = {
      server_root_path: "/",
      proxy_base_url: null,
      auto_redirect_to_sso: false,
      admin_ui_disabled: false,
      sso_configured: false,
    };

    const setSessionCookie = (claims: Record<string, unknown>) => {
      decodeTokenMock.mockReturnValue(claims);
      checkTokenValidityMock.mockReturnValue(true);
      const token = createJwt(claims);
      document.cookie = `token=${token}; path=/;`;
      return token;
    };

    it("uses the role the server resolved, not the role claimed in the cookie", async () => {
      getUiConfigMock.mockResolvedValue(uiConfig);
      setSessionCookie({ key: "internal-key", user_id: "internal-1", user_role: "proxy_admin" });
      serveCurrentUser({
        user_id: "internal-1",
        user_email: "internal@example.com",
        user_role: "internal_user",
      });

      const { result } = renderHook(() => useAuthorized(), { wrapper });

      await waitFor(() => {
        expect(result.current.userRole).toBe("Internal User");
      });
      expect(result.current.userRoleLabel).toBe("Internal User");
      expect(result.current.isViewOnly).toBe(false);
      expect(result.current.userEmail).toBe("internal@example.com");
      expect(replaceMock).not.toHaveBeenCalled();
    });

    it("stays loading with no role until the server answers", async () => {
      getUiConfigMock.mockResolvedValue(uiConfig);
      setSessionCookie({ key: "admin-claim-key", user_id: "u-1", user_role: "proxy_admin" });
      stallCurrentUser();

      const { result } = renderHook(() => ({ auth: useAuthorized(), uiConfig: useUIConfig() }), { wrapper });

      await waitFor(() => {
        expect(result.current.uiConfig.isSuccess).toBe(true);
      });
      expect(result.current.auth.isLoading).toBe(true);
      expect(result.current.auth.userRole).toBe("Undefined Role");
      expect(replaceMock).not.toHaveBeenCalled();
    });

    it("waits for backend discovery before looking up the current user", async () => {
      let finishDiscovery!: (config: typeof uiConfig) => void;
      getUiConfigMock.mockReturnValue(new Promise((resolve) => (finishDiscovery = resolve)));
      setSessionCookie({ key: "discovery-key", user_id: "u-9", user_role: "internal_user" });
      serveCurrentUser({ user_id: "u-9", user_email: null, user_role: "internal_user" });

      const { result } = renderHook(() => useAuthorized(), { wrapper });

      await waitFor(() => expect(getUiConfigMock).toHaveBeenCalled());
      await act(() => new Promise((resolve) => setTimeout(resolve, 50)));
      expect(currentUserRequests()).toHaveLength(0);
      expect(result.current.isLoading).toBe(true);

      await act(async () => finishDiscovery(uiConfig));

      await waitFor(() => expect(result.current.userId).toBe("u-9"));
      expect(currentUserRequests()).toHaveLength(1);
      expect(replaceMock).not.toHaveBeenCalled();
    });

    it("keeps an identity the page already loaded while backend discovery reruns", async () => {
      getUiConfigMock.mockReturnValue(new Promise(() => {}));
      setSessionCookie({ key: "cached-key", user_id: "u-7", user_role: "proxy_admin" });
      serveCurrentUser({ user_id: "u-7", user_email: null, user_role: "internal_user" });
      const queryClient = createQueryClient();
      const sharedWrapper = ({ children }: { children: React.ReactNode }) =>
        React.createElement(QueryClientProvider, { client: queryClient }, children);

      const { result: layoutLookup } = renderHook(() => useCurrentUser("cached-key"), { wrapper: sharedWrapper });
      await waitFor(() => expect(layoutLookup.current.isSuccess).toBe(true));

      const { result } = renderHook(() => useAuthorized(), { wrapper: sharedWrapper });

      expect(result.current.userId).toBe("u-7");
      expect(result.current.userRole).toBe("Internal User");
      expect(currentUserRequests()).toHaveLength(1);
    });

    it("clears the cookie and redirects to login when the current user cannot be loaded", async () => {
      getUiConfigMock.mockResolvedValue(uiConfig);
      setSessionCookie({ key: "orphan-key", user_id: "u-1", user_role: "internal_user" });
      failCurrentUser(404);

      renderHook(() => useAuthorized(), { wrapper });

      await waitFor(
        () => {
          expect(replaceMock).toHaveBeenCalledWith("http://proxy.example/ui/login/");
        },
        { timeout: 4000 },
      );
      expect(clearTokenCookiesMock).toHaveBeenCalled();
      expect(currentUserRequests()).toHaveLength(2);
    });

    it("refetches the identity when a different session key logs in", async () => {
      getUiConfigMock.mockResolvedValue(uiConfig);
      serveCurrentUser((request) =>
        request.headers.get("Authorization") === "Bearer first-key"
          ? { user_id: "first", user_email: null, user_role: "proxy_admin" }
          : { user_id: "second", user_email: null, user_role: "internal_user" },
      );
      const queryClient = createQueryClient();
      const sharedClientWrapper = ({ children }: { children: React.ReactNode }) =>
        React.createElement(QueryClientProvider, { client: queryClient }, children);
      setSessionCookie({ key: "first-key", user_id: "first" });

      const { result, rerender } = renderHook(() => useAuthorized(), { wrapper: sharedClientWrapper });
      await waitFor(() => {
        expect(result.current.userId).toBe("first");
      });

      setSessionCookie({ key: "second-key", user_id: "second" });
      act(() => rerender());

      await waitFor(() => {
        expect(result.current.userId).toBe("second");
      });
      expect(result.current.userRole).toBe("Internal User");
    });

    it.each([
      { cookieRole: "internal_user", serverRole: "proxy_admin_viewer", isViewOnly: true },
      { cookieRole: "proxy_admin_viewer", serverRole: "proxy_admin", isViewOnly: false },
    ])("derives view-only from the server role ($serverRole), not the cookie ($cookieRole)", async (scenario) => {
      getUiConfigMock.mockResolvedValue(uiConfig);
      setSessionCookie({ key: `view-only-${scenario.serverRole}`, user_id: "u-1", user_role: scenario.cookieRole });
      serveCurrentUser({ user_id: "u-1", user_email: null, user_role: scenario.serverRole });

      const { result } = renderHook(() => useAuthorized(), { wrapper });

      await waitFor(() => {
        expect(result.current.userId).toBe("u-1");
      });
      expect(result.current.isViewOnly).toBe(scenario.isViewOnly);
    });

    it("does not refetch the identity when a consumer mounts after it has loaded", async () => {
      getUiConfigMock.mockResolvedValue(uiConfig);
      setSessionCookie({ key: "late-key", user_id: "u-1" });
      serveCurrentUser({ user_id: "u-1", user_email: null, user_role: "internal_user" });
      const queryClient = createQueryClient();
      const sharedClientWrapper = ({ children }: { children: React.ReactNode }) =>
        React.createElement(QueryClientProvider, { client: queryClient }, children);

      const first = renderHook(() => useAuthorized(), { wrapper: sharedClientWrapper });
      await waitFor(() => {
        expect(first.result.current.userId).toBe("u-1");
      });
      const second = renderHook(() => useAuthorized(), { wrapper: sharedClientWrapper });

      expect(second.result.current.userId).toBe("u-1");
      await act(async () => {});
      expect(currentUserRequests()).toHaveLength(1);
    });

    it("fetches the identity once for many consumers on the page", async () => {
      getUiConfigMock.mockResolvedValue(uiConfig);
      setSessionCookie({ key: "shared-key", user_id: "u-1" });
      serveCurrentUser({ user_id: "u-1", user_email: null, user_role: "internal_user" });
      const queryClient = createQueryClient();
      const sharedClientWrapper = ({ children }: { children: React.ReactNode }) =>
        React.createElement(QueryClientProvider, { client: queryClient }, children);

      const { result } = renderHook(() => [useAuthorized(), useAuthorized(), useAuthorized()], {
        wrapper: sharedClientWrapper,
      });

      await waitFor(() => {
        expect(result.current.map((auth) => auth.userId)).toEqual(["u-1", "u-1", "u-1"]);
      });
      expect(currentUserRequests()).toHaveLength(1);
    });
  });
});
