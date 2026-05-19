import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import LoginPage from "./LoginPage";

const mockPush = vi.fn();
const mockReplace = vi.fn();

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
  getCookie: vi.fn(),
}));

vi.mock("@/utils/jwtUtils", () => ({
  isJwtExpired: vi.fn(),
}));

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost:4000"),
  };
});

vi.mock("@/app/(dashboard)/hooks/login/useLogin", () => ({
  useLogin: vi.fn(() => ({
    mutate: vi.fn(),
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
import { getCookie } from "@/utils/cookieUtils";
import { isJwtExpired } from "@/utils/jwtUtils";

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
  beforeEach(() => {
    vi.clearAllMocks();
    mockPush.mockClear();
    mockReplace.mockClear();
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
    (getCookie as ReturnType<typeof vi.fn>).mockReturnValue(null);

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

  it("should call router.replace to dashboard when jwt is valid", async () => {
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
    (getCookie as ReturnType<typeof vi.fn>).mockReturnValue(validToken);
    (isJwtExpired as ReturnType<typeof vi.fn>).mockReturnValue(false);

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <LoginPage />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith("/ui");
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
    (getCookie as ReturnType<typeof vi.fn>).mockReturnValue(invalidToken);
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
    (getCookie as ReturnType<typeof vi.fn>).mockReturnValue(invalidToken);
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
    (getCookie as ReturnType<typeof vi.fn>).mockReturnValue(validToken);
    (isJwtExpired as ReturnType<typeof vi.fn>).mockReturnValue(false);

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <LoginPage />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith("/ui");
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
    (getCookie as ReturnType<typeof vi.fn>).mockReturnValue(null);

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
    (getCookie as ReturnType<typeof vi.fn>).mockReturnValue(null);
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
    (getCookie as ReturnType<typeof vi.fn>).mockReturnValue(null);
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

  describe("URL ?token= legacy path is rejected (security regression test)", () => {
    const originalLocation = window.location;

    beforeEach(() => {
      Object.defineProperty(window, "location", {
        value: {
          ...originalLocation,
          href: "http://localhost:3000/ui/login?token=attacker.jwt.value",
          pathname: "/ui/login",
          search: "?token=attacker.jwt.value",
        },
        writable: true,
      });
      document.cookie =
        "token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; SameSite=Lax";
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
      (getCookie as ReturnType<typeof vi.fn>).mockReturnValue(null);
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
      expect(mockReplace).not.toHaveBeenCalledWith("/ui/?login=success");
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
      (getCookie as ReturnType<typeof vi.fn>).mockReturnValue("legitimate-session-jwt");
      (isJwtExpired as ReturnType<typeof vi.fn>).mockReturnValue(false);

      const queryClient = createQueryClient();
      render(
        <QueryClientProvider client={queryClient}>
          <LoginPage />
        </QueryClientProvider>,
      );

      await waitFor(() => {
        expect(mockReplace).toHaveBeenCalledWith("/ui");
      });

      expect(document.cookie).not.toContain("token=attacker.jwt.value");
      expect(mockReplace).not.toHaveBeenCalledWith("/ui/?login=success");
    });
  });
});
