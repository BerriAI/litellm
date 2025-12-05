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
      data: { auto_redirect_to_sso: false, server_root_path: "/", proxy_base_url: null },
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
      data: { auto_redirect_to_sso: false, server_root_path: "/", proxy_base_url: null },
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
      expect(mockReplace).toHaveBeenCalledWith("http://localhost:4000/ui");
    });
  });

  it("should call router.push to SSO when jwt is invalid and auto_redirect_to_sso is true", async () => {
    const invalidToken = "invalid-token";
    (useUIConfig as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { auto_redirect_to_sso: true, server_root_path: "/", proxy_base_url: null },
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
      data: { auto_redirect_to_sso: false, server_root_path: "/", proxy_base_url: null },
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
      data: { auto_redirect_to_sso: true, server_root_path: "/", proxy_base_url: null },
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
      expect(mockReplace).toHaveBeenCalledWith("http://localhost:4000/ui");
    });

    expect(mockPush).not.toHaveBeenCalled();
  });
});
