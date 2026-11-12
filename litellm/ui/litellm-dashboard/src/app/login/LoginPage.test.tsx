import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import LoginPage from "./LoginPage";

const mockPush = vi.fn();
const mockReplace = vi.fn();
const mockAssignDocument = vi.fn();
const mockReplaceDocument = vi.fn();
const mockMutate = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: vi.fn(() => ({
    push: mockPush,
    replace: mockReplace,
  })),
}));

vi.mock("@/app/(dashboard)/hooks/uiConfig/useUIConfig", () => ({
  useUIConfig: vi.fn(),
}));

vi.mock("@/utils/cookieUtils", () => ({
  clearTokenCookies: vi.fn(),
  getCookieFromDocument: vi.fn(),
}));

vi.mock("@/utils/jwtUtils", () => ({
  isJwtExpired: vi.fn(),
}));

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost:4000"),
    exchangeLoginCode: vi.fn(),
  };
});

vi.mock("@/app/(dashboard)/hooks/login/useLogin", () => ({
  useLogin: vi.fn(() => ({
    mutate: mockMutate,
    isPending: false,
    error: null,
  })),
}));

vi.mock("@/hooks/useWorker", () => ({
  useWorker: vi.fn(() => ({
    isControlPlane: false,
    workers: [],
    selectedWorkerId: null,
    selectedWorker: null,
    selectWorker: vi.fn(),
    disconnectFromWorker: vi.fn(),
  })),
}));

import { useUIConfig } from "@/app/(dashboard)/hooks/uiConfig/useUIConfig";
import { getCookieFromDocument } from "@/utils/cookieUtils";
import { isJwtExpired } from "@/utils/jwtUtils";
import { exchangeLoginCode } from "@/components/networking";

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

describe("LoginPage", () => {
  const originalLocation = window.location;

  beforeEach(() => {
    vi.clearAllMocks();
    mockPush.mockClear();
    mockReplace.mockClear();
    mockMutate.mockReset();
    Object.defineProperty(window, "location", {
      value: { ...originalLocation, assign: mockAssignDocument, replace: mockReplaceDocument },
      writable: true,
    });
    document.cookie = "litellm_return_url=; Max-Age=0; Path=/";
  });

  afterEach(() => {
    Object.defineProperty(window, "location", { value: originalLocation, writable: true });
    document.cookie = "litellm_return_url=; Max-Age=0; Path=/";
  });

  it("should render", async () => {
    (useUIConfig as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        auto_redirect_to_sso: false,
        server_root_path: "/",
        proxy_base_url: null,
        sso_configured: false,
      },
      isLoading: false,
    });
    (getCookieFromDocument as ReturnType<typeof vi.fn>).mockReturnValue(null);

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <LoginPage />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "Login" })).toBeInTheDocument();
    });
  });

  it("loads a fresh dashboard document when jwt is valid", async () => {
    const validToken = "valid-token";
    (useUIConfig as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        auto_redirect_to_sso: false,
        server_root_path: "/",
        proxy_base_url: null,
        sso_configured: false,
      },
      isLoading: false,
    });
    (getCookieFromDocument as ReturnType<typeof vi.fn>).mockReturnValue(validToken);
    (isJwtExpired as ReturnType<typeof vi.fn>).mockReturnValue(false);

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <LoginPage />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(mockReplaceDocument).toHaveBeenCalledWith(new URL("/ui", window.location.origin).href);
    });
  });

  it("should call router.push to SSO when jwt is invalid and auto_redirect_to_sso is true", async () => {
    const invalidToken = "invalid-token";
    (useUIConfig as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        auto_redirect_to_sso: true,
        server_root_path: "/",
        proxy_base_url: null,
        sso_configured: true,
      },
      isLoading: false,
    });
    (getCookieFromDocument as ReturnType<typeof vi.fn>).mockReturnValue(invalidToken);
    (isJwtExpired as ReturnType<typeof vi.fn>).mockReturnValue(true);

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <LoginPage />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith("http://localhost:4000/sso/key/generate");
    });
  });

  it("should not call router when jwt is invalid and auto_redirect_to_sso is false", async () => {
    const invalidToken = "invalid-token";
    (useUIConfig as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        auto_redirect_to_sso: false,
        server_root_path: "/",
        proxy_base_url: null,
        sso_configured: false,
      },
      isLoading: false,
    });
    (getCookieFromDocument as ReturnType<typeof vi.fn>).mockReturnValue(invalidToken);
    (isJwtExpired as ReturnType<typeof vi.fn>).mockReturnValue(true);

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <LoginPage />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "Login" })).toBeInTheDocument();
    });

    expect(mockPush).not.toHaveBeenCalled();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("should send user to dashboard when jwt is valid even if auto_redirect_to_sso is true", async () => {
    const validToken = "valid-token";
    (useUIConfig as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        auto_redirect_to_sso: true,
        server_root_path: "/",
        proxy_base_url: null,
        sso_configured: true,
      },
      isLoading: false,
    });
    (getCookieFromDocument as ReturnType<typeof vi.fn>).mockReturnValue(validToken);
    (isJwtExpired as ReturnType<typeof vi.fn>).mockReturnValue(false);

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <LoginPage />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(mockReplaceDocument).toHaveBeenCalledWith(new URL("/ui", window.location.origin).href);
    });

    expect(mockPush).not.toHaveBeenCalled();
  });

  it("should show alert when admin_ui_disabled is true", async () => {
    (useUIConfig as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        admin_ui_disabled: true,
        server_root_path: "/",
        proxy_base_url: null,
        sso_configured: false,
      },
      isLoading: false,
    });
    (getCookieFromDocument as ReturnType<typeof vi.fn>).mockReturnValue(null);

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <LoginPage />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
      expect(screen.getByText("Admin UI Disabled")).toBeInTheDocument();
    });

    expect(mockPush).not.toHaveBeenCalled();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("should show Login with SSO button when sso_configured is true", async () => {
    (useUIConfig as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        auto_redirect_to_sso: false,
        server_root_path: "/",
        proxy_base_url: null,
        sso_configured: true,
      },
      isLoading: false,
    });
    (getCookieFromDocument as ReturnType<typeof vi.fn>).mockReturnValue(null);
    (isJwtExpired as ReturnType<typeof vi.fn>).mockReturnValue(true);

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <LoginPage />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "Login" })).toBeInTheDocument();
    });

    expect(screen.getByRole("button", { name: "Login with SSO" })).toBeInTheDocument();
  });

  it("should show disabled Login with SSO button with popover when sso_configured is false", async () => {
    (useUIConfig as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        auto_redirect_to_sso: false,
        server_root_path: "/",
        proxy_base_url: null,
        sso_configured: false,
      },
      isLoading: false,
    });
    (getCookieFromDocument as ReturnType<typeof vi.fn>).mockReturnValue(null);
    (isJwtExpired as ReturnType<typeof vi.fn>).mockReturnValue(true);

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <LoginPage />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "Login" })).toBeInTheDocument();
    });

    const ssoButton = screen.getByRole("button", { name: "Login with SSO" });
    expect(ssoButton).toBeInTheDocument();
    expect(ssoButton).toBeDisabled();
  });

  describe("post-login document navigation", () => {
    beforeEach(() => {
      vi.mocked(useUIConfig).mockReturnValue({
        data: { auto_redirect_to_sso: false, server_root_path: "/", proxy_base_url: null, sso_configured: false },
        isLoading: false,
      } as ReturnType<typeof useUIConfig>);
      vi.mocked(getCookieFromDocument).mockReturnValue(null);
      vi.mocked(isJwtExpired).mockReturnValue(true);
    });

    function renderLogin() {
      return render(
        <QueryClientProvider client={createQueryClient()}>
          <LoginPage />
        </QueryClientProvider>,
      );
    }

    it.each([
      [null, "/ui/?login=success"],
      ["/ui/teams/?tab=all", "/ui/teams/?tab=all"],
      ["https://untrusted.example/ui/", "/ui/?login=success"],
      ["/\\attacker.example/ui/", "/ui/?login=success"],
      ["/\t/attacker.example/ui/", "/ui/?login=success"],
      ["javascript://localhost/%0Aalert(1)", "/ui/?login=success"],
    ])("loads the dashboard document after password login with return URL %s", async (returnUrl, expectedUrl) => {
      if (returnUrl) document.cookie = `litellm_return_url=${encodeURIComponent(returnUrl)}; Path=/`;
      mockMutate.mockImplementation((_input, options) => options.onSuccess({ redirect_url: "/ui/?login=success" }));
      renderLogin();
      await screen.findByRole("heading", { name: "Login" });
      fireEvent.change(screen.getByLabelText("Username"), { target: { value: "admin" } });
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "test-password" } });
      fireEvent.click(screen.getByRole("button", { name: "Login", exact: true }));

      await waitFor(() =>
        expect(mockAssignDocument).toHaveBeenCalledWith(new URL(expectedUrl, window.location.origin).href),
      );
      expect(mockPush).not.toHaveBeenCalled();
    });

    it("replaces the login document after SSO code exchange succeeds", async () => {
      window.location.search = "?code=single-use-code";
      vi.mocked(exchangeLoginCode).mockResolvedValue("session-jwt");
      renderLogin();

      await waitFor(() => expect(mockReplaceDocument).toHaveBeenCalledWith("/ui/?login=success"));
      expect(exchangeLoginCode).toHaveBeenCalledWith("single-use-code", null);
      expect(mockReplace).not.toHaveBeenCalled();
    });

    it.each(["https://untrusted.example/ui/", "javascript://localhost/%0Aalert(1)"])(
      "rejects an unsafe server-provided login redirect %j",
      async (redirectUrl) => {
        mockMutate.mockImplementation((_input, options) => options.onSuccess({ redirect_url: redirectUrl }));
        renderLogin();
        await screen.findByRole("heading", { name: "Login" });
        fireEvent.change(screen.getByLabelText("Username"), { target: { value: "admin" } });
        fireEvent.change(screen.getByLabelText("Password"), { target: { value: "test-password" } });
        fireEvent.click(screen.getByRole("button", { name: "Login", exact: true }));

        await waitFor(() => expect(mockAssignDocument).toHaveBeenCalledWith("/ui"));
      },
    );

    it("replaces the login document with a valid return URL for an existing session", async () => {
      document.cookie = `litellm_return_url=${encodeURIComponent("/ui/teams/")}; Path=/`;
      vi.mocked(getCookieFromDocument).mockReturnValue("valid-session");
      vi.mocked(isJwtExpired).mockReturnValue(false);
      renderLogin();

      await waitFor(() =>
        expect(mockReplaceDocument).toHaveBeenCalledWith(new URL("/ui/teams/", window.location.origin).href),
      );
      expect(mockReplace).not.toHaveBeenCalled();
    });

    it.each(["/\\attacker.example/ui/", "javascript://localhost/%0Aalert(1)"])(
      "keeps an existing session on the dashboard for unsafe return URL %j",
      async (returnUrl) => {
        window.location.search = `?redirect_to=${encodeURIComponent(returnUrl)}`;
        vi.mocked(getCookieFromDocument).mockReturnValue("valid-session");
        vi.mocked(isJwtExpired).mockReturnValue(false);
        renderLogin();

        await waitFor(() =>
          expect(mockReplaceDocument).toHaveBeenCalledWith(new URL("/ui", window.location.origin).href),
        );
      },
    );
  });

  describe("URL ?token= legacy path is rejected (security regression test)", () => {
    const originalLocation = window.location;

    beforeEach(() => {
      Object.defineProperty(window, "location", {
        value: {
          ...window.location,
          href: "http://localhost:3000/ui/login?token=attacker.jwt.value",
          pathname: "/ui/login",
          search: "?token=attacker.jwt.value",
        },
        writable: true,
      });
      document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; SameSite=Lax";
    });

    afterEach(() => {
      Object.defineProperty(window, "location", {
        value: originalLocation,
        writable: true,
      });
    });

    it("must not set a token cookie or redirect to /ui/?login=success when ?token= is in the URL", async () => {
      (useUIConfig as ReturnType<typeof vi.fn>).mockReturnValue({
        data: {
          auto_redirect_to_sso: false,
          server_root_path: "/",
          proxy_base_url: null,
          sso_configured: false,
        },
        isLoading: false,
      });
      (getCookieFromDocument as ReturnType<typeof vi.fn>).mockReturnValue(null);
      (isJwtExpired as ReturnType<typeof vi.fn>).mockReturnValue(false);

      const queryClient = createQueryClient();
      render(
        <QueryClientProvider client={queryClient}>
          <LoginPage />
        </QueryClientProvider>,
      );

      await waitFor(() => {
        expect(screen.getByRole("heading", { name: "Login" })).toBeInTheDocument();
      });

      expect(document.cookie).not.toContain("token=attacker.jwt.value");
      expect(mockReplaceDocument).not.toHaveBeenCalledWith("/ui/?login=success");
    });

    it("must not overwrite an existing valid session cookie when ?token= is in the URL", async () => {
      (useUIConfig as ReturnType<typeof vi.fn>).mockReturnValue({
        data: {
          auto_redirect_to_sso: false,
          server_root_path: "/",
          proxy_base_url: null,
          sso_configured: false,
        },
        isLoading: false,
      });
      (getCookieFromDocument as ReturnType<typeof vi.fn>).mockReturnValue("legitimate-session-jwt");
      (isJwtExpired as ReturnType<typeof vi.fn>).mockReturnValue(false);

      const queryClient = createQueryClient();
      render(
        <QueryClientProvider client={queryClient}>
          <LoginPage />
        </QueryClientProvider>,
      );

      await waitFor(() => {
        expect(mockReplaceDocument).toHaveBeenCalledWith(new URL("/ui", window.location.origin).href);
      });

      expect(document.cookie).not.toContain("token=attacker.jwt.value");
      expect(mockReplaceDocument).not.toHaveBeenCalledWith("/ui/?login=success");
    });
  });
});
