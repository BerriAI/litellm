import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import LoginPage from "./LoginPage";

const mockPush = vi.fn();
const mockReplace = vi.fn();
const mockMutate = vi.fn();
const mockSelectWorker = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: vi.fn(() => ({ push: mockPush, replace: mockReplace })),
}));

vi.mock("@/app/(dashboard)/hooks/uiConfig/useUIConfig", () => ({
  useUIConfig: vi.fn(),
}));

vi.mock("@/utils/cookieUtils", () => ({
  clearTokenCookies: vi.fn(),
  getCookieFromDocument: vi.fn(() => null),
}));

vi.mock("@/utils/jwtUtils", () => ({
  isJwtExpired: vi.fn(() => true),
}));

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    getProxyBaseUrl: vi.fn(() => "http://localhost:4000"),
    switchToWorkerUrl: vi.fn(),
    exchangeLoginCode: vi.fn(),
  };
});

vi.mock("@/app/(dashboard)/hooks/login/useLogin", () => ({
  useLogin: vi.fn(() => ({ mutate: mockMutate, isPending: false, error: null })),
}));

vi.mock("@/hooks/useWorker", () => ({
  useWorker: vi.fn(() => ({
    isControlPlane: false,
    workers: WORKERS,
    selectedWorkerId: null,
    selectedWorker: null,
    selectWorker: mockSelectWorker,
    disconnectFromWorker: vi.fn(),
  })),
}));

import { useUIConfig } from "@/app/(dashboard)/hooks/uiConfig/useUIConfig";
import { switchToWorkerUrl } from "@/components/networking";

let WORKERS: { worker_id: string; name: string; url: string }[] = [];

const renderLoginPage = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <LoginPage />
    </QueryClientProvider>,
  );
};

const setSearch = (search: string) => {
  Object.defineProperty(window, "location", {
    value: { ...window.location, pathname: "/ui/login", search, href: `http://localhost:3000/ui/login${search}` },
    writable: true,
  });
};

const originalLocation = window.location;

describe("LoginPage submit payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    WORKERS = [];
    setSearch("");
    (useUIConfig as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        auto_redirect_to_sso: false,
        server_root_path: "/",
        proxy_base_url: null,
        sso_configured: false,
      },
      isLoading: false,
    });
  });

  afterEach(() => {
    Object.defineProperty(window, "location", { value: originalLocation, writable: true });
  });

  it("sends exactly {username, password, useV3:false} when the Login button is clicked", async () => {
    const user = userEvent.setup();
    renderLoginPage();
    await screen.findByRole("heading", { name: "Login" });

    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "admin" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "sk-1234" } });
    await user.click(screen.getByRole("button", { name: "Login" }));

    await waitFor(() => expect(mockMutate).toHaveBeenCalledTimes(1));
    expect(mockMutate.mock.calls[0][0]).toStrictEqual({ username: "admin", password: "sk-1234", useV3: false });
    expect(switchToWorkerUrl).not.toHaveBeenCalled();
  });

  it("submits on Enter from the password field", async () => {
    const user = userEvent.setup();
    renderLoginPage();
    await screen.findByRole("heading", { name: "Login" });

    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "admin" } });
    await user.type(screen.getByLabelText("Password"), "sk-1234{Enter}");

    await waitFor(() => expect(mockMutate).toHaveBeenCalledTimes(1));
    expect(mockMutate.mock.calls[0][0]).toStrictEqual({ username: "admin", password: "sk-1234", useV3: false });
  });

  it("blocks submit and shows both required messages when the fields are empty", async () => {
    const user = userEvent.setup();
    renderLoginPage();
    await screen.findByRole("heading", { name: "Login" });

    await user.click(screen.getByRole("button", { name: "Login" }));

    expect(await screen.findByText("Please enter your username")).toBeInTheDocument();
    expect(screen.getByText("Please enter your password")).toBeInTheDocument();
    expect(mockMutate).not.toHaveBeenCalled();
  });

  it("sends useV3:true for a worker chosen from the Worker picker", async () => {
    const user = userEvent.setup();
    WORKERS = [
      { worker_id: "worker-a", name: "Worker A", url: "http://worker-a:4000" },
      { worker_id: "worker-b", name: "Worker B", url: "http://worker-b:4000" },
    ];
    (useUIConfig as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        auto_redirect_to_sso: false,
        server_root_path: "/",
        proxy_base_url: null,
        sso_configured: false,
        is_control_plane: true,
      },
      isLoading: false,
    });

    renderLoginPage();
    await screen.findByRole("heading", { name: "Login" });

    await user.click(screen.getAllByRole("combobox")[0]);
    await user.click(await screen.findByText("Worker B"));
    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "admin" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "sk-1234" } });
    await user.click(screen.getByRole("button", { name: "Login" }));

    await waitFor(() => expect(mockMutate).toHaveBeenCalledTimes(1));
    expect(mockMutate.mock.calls[0][0]).toStrictEqual({ username: "admin", password: "sk-1234", useV3: true });
    expect(switchToWorkerUrl).toHaveBeenCalledWith("http://worker-b:4000");
  });

  it("switches to the selected worker url and sends useV3:true when a worker is preselected", async () => {
    const user = userEvent.setup();
    WORKERS = [{ worker_id: "worker-a", name: "Worker A", url: "http://worker-a:4000" }];
    setSearch("?worker=worker-a");
    (useUIConfig as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        auto_redirect_to_sso: false,
        server_root_path: "/",
        proxy_base_url: null,
        sso_configured: false,
        is_control_plane: true,
      },
      isLoading: false,
    });

    renderLoginPage();
    await screen.findByRole("heading", { name: "Login" });

    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "admin" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "sk-1234" } });
    await user.click(screen.getByRole("button", { name: "Login" }));

    await waitFor(() => expect(mockMutate).toHaveBeenCalledTimes(1));
    expect(mockMutate.mock.calls[0][0]).toStrictEqual({ username: "admin", password: "sk-1234", useV3: true });
    expect(switchToWorkerUrl).toHaveBeenCalledWith("http://worker-a:4000");
  });
});
