import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import LoginPage from "./LoginPage";
import { LanguageProvider } from "@/contexts/LanguageContext";
import type { Language } from "@/i18n/language";

const mockPush = vi.fn();
const mockReplace = vi.fn();
const mockMutate = vi.fn();
const mockSelectWorker = vi.fn();
let mockLoginError: Error | null = null;

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
  useLogin: vi.fn(() => ({ mutate: mockMutate, isPending: false, error: mockLoginError })),
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

const renderLoginPage = (language: Language = "en") => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <LanguageProvider defaultLanguage={language} userId={null} storageScope="login-test">
        <LoginPage />
      </LanguageProvider>
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
    localStorage.clear();
    WORKERS = [];
    mockLoginError = null;
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
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "sk-9876" } });
    await user.click(screen.getByRole("button", { name: "Login" }));

    await waitFor(() => expect(mockMutate).toHaveBeenCalledTimes(1));
    expect(mockMutate.mock.calls[0][0]).toStrictEqual({ username: "admin", password: "sk-9876", useV3: false });
    expect(switchToWorkerUrl).not.toHaveBeenCalled();
  });

  it("中文界面显示本地校验消息，提交凭据保持原值并可切换回英文", async () => {
    const user = userEvent.setup();
    renderLoginPage("zh-CN");
    await screen.findByRole("heading", { name: "登录" });
    await user.click(screen.getByRole("button", { name: "登录", exact: true }));
    expect(await screen.findByText("请输入用户名")).toBeVisible();
    expect(mockMutate).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("用户名"), { target: { value: "Settings" } });
    fireEvent.change(screen.getByLabelText("密码", { exact: true }), { target: { value: "Home" } });
    await user.click(screen.getByRole("button", { name: "登录", exact: true }));
    await waitFor(() => expect(mockMutate).toHaveBeenCalledTimes(1));
    expect(mockMutate.mock.calls[0][0]).toStrictEqual({ username: "Settings", password: "Home", useV3: false });
    await user.selectOptions(screen.getByRole("combobox", { name: "界面语言" }), "en");
    expect(screen.getByRole("heading", { name: "Login" })).toBeVisible();
    expect(screen.getByLabelText("Username")).toHaveValue("Settings");
    expect(screen.getByLabelText("Password", { exact: true })).toHaveValue("Home");
  });

  it("submits on Enter from the password field", async () => {
    const user = userEvent.setup();
    renderLoginPage();
    await screen.findByRole("heading", { name: "Login" });

    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "admin" } });
    await user.type(screen.getByLabelText("Password"), "sk-9876{Enter}");

    await waitFor(() => expect(mockMutate).toHaveBeenCalledTimes(1));
    expect(mockMutate.mock.calls[0][0]).toStrictEqual({ username: "admin", password: "sk-9876", useV3: false });
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

  it("已显示的登录校验随语言切换且不会登录、跳转或清空用户名", async () => {
    renderLoginPage();
    await screen.findByRole("heading", { name: "Login" });
    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "Settings" } });
    fireEvent.click(screen.getByRole("button", { name: "Login", exact: true }));
    expect(await screen.findByText("Please enter your password")).toBeVisible();

    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    expect(await screen.findByText("请输入密码")).toBeVisible();
    expect(screen.getByLabelText("用户名")).toHaveValue("Settings");
    expect(screen.queryByText("请输入用户名")).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(await screen.findByText("Please enter your password")).toBeVisible();
    expect(mockMutate).not.toHaveBeenCalled();
    expect(mockPush).not.toHaveBeenCalled();
    expect(mockReplace).not.toHaveBeenCalled();
    expect(switchToWorkerUrl).not.toHaveBeenCalled();
  });

  it("登录接口的同名错误在切换语言后仍显示原文", async () => {
    // 网络层错误与字段校验来源不同，即使恰好同名也不能翻译。
    mockLoginError = new Error("Please enter your password");
    renderLoginPage();
    await screen.findByRole("heading", { name: "Login" });
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    expect(screen.getByText("Please enter your password")).toBeVisible();
    expect(screen.queryByText("请输入密码")).not.toBeInTheDocument();
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
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "sk-9876" } });
    await user.click(screen.getByRole("button", { name: "Login" }));

    await waitFor(() => expect(mockMutate).toHaveBeenCalledTimes(1));
    expect(mockMutate.mock.calls[0][0]).toStrictEqual({ username: "admin", password: "sk-9876", useV3: true });
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
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "sk-9876" } });
    await user.click(screen.getByRole("button", { name: "Login" }));

    await waitFor(() => expect(mockMutate).toHaveBeenCalledTimes(1));
    expect(mockMutate.mock.calls[0][0]).toStrictEqual({ username: "admin", password: "sk-9876", useV3: true });
    expect(switchToWorkerUrl).toHaveBeenCalledWith("http://worker-a:4000");
  });
});
