import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import I18nProvider from "@/i18n/I18nProvider";
import i18n from "@/i18n/i18n";
import { LANGUAGE_STORAGE_KEY, nativeName } from "@/i18n/languages";
import LoginPage from "./LoginPage";

const mockPush = vi.fn();
const mockReplace = vi.fn();
const mockMutate = vi.fn();
const mockSelectWorker = vi.fn();
const mockReload = vi.fn();

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
    <I18nProvider>
      <QueryClientProvider client={queryClient}>
        <LoginPage />
      </QueryClientProvider>
    </I18nProvider>,
  );
};

const setSearch = (search: string) => {
  Object.defineProperty(window, "location", {
    value: {
      ...window.location,
      pathname: "/ui/login",
      search,
      href: `http://localhost:3000/ui/login${search}`,
      reload: mockReload,
    },
    writable: true,
  });
};

const originalLocation = window.location;

describe("LoginPage submit payload", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    localStorage.removeItem(LANGUAGE_STORAGE_KEY);
    await i18n.changeLanguage("en");
    document.documentElement.lang = "en";
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

  afterEach(async () => {
    Object.defineProperty(window, "location", { value: originalLocation, writable: true });
    localStorage.removeItem(LANGUAGE_STORAGE_KEY);
    await i18n.changeLanguage("en");
    document.documentElement.lang = "en";
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

    await user.click(screen.getByRole("combobox", { name: "Worker" }));
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

  it("restores Chinese login labels and validation after choosing the language before signing in", async () => {
    const user = userEvent.setup();
    const page = renderLoginPage();
    await screen.findByRole("heading", { name: "Login" });

    await user.click(screen.getByRole("combobox", { name: /Language/ }));
    await user.click(await screen.findByRole("option", { name: new RegExp(nativeName("zh-CN")) }));

    expect(localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe("zh-CN");
    expect(mockReload).toHaveBeenCalledTimes(1);
    page.unmount();
    renderLoginPage();

    expect(await screen.findByRole("heading", { name: "登录" })).toBeInTheDocument();
    expect(screen.getByLabelText("用户名")).toHaveAttribute("placeholder", "请输入用户名");
    expect(screen.getByLabelText("密码")).toHaveAttribute("placeholder", "请输入密码");
    expect(screen.getByText("MASTER_KEY")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "查看文档" })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/proxy/ui",
    );
    expect(document.documentElement.lang).toBe("zh-CN");
    await user.click(screen.getByRole("button", { name: "登录", exact: true }));
    expect(await screen.findByText("请输入用户名")).toBeInTheDocument();
    expect(screen.getByText("请输入密码")).toBeInTheDocument();
    expect(mockMutate).not.toHaveBeenCalled();
  });

  it("submits the same credential payload from a Chinese login page", async () => {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, "zh-CN");
    renderLoginPage();
    await screen.findByRole("heading", { name: "登录" });

    fireEvent.change(screen.getByLabelText("用户名"), { target: { value: "admin" } });
    fireEvent.change(screen.getByLabelText("密码"), { target: { value: "sk-9876" } });
    await userEvent.click(screen.getByRole("button", { name: "登录", exact: true }));

    await waitFor(() => expect(mockMutate).toHaveBeenCalledTimes(1));
    expect(mockMutate.mock.calls[0][0]).toStrictEqual({ username: "admin", password: "sk-9876", useV3: false });
  });

  it("lets a Chinese disabled-admin page save English without exposing the login form", async () => {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, "zh-CN");
    vi.mocked(useUIConfig).mockReturnValue({ data: { admin_ui_disabled: true }, isLoading: false } as ReturnType<
      typeof useUIConfig
    >);
    renderLoginPage();

    expect(await screen.findByText("管理界面已禁用")).toBeInTheDocument();
    expect(screen.queryByLabelText("用户名")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("combobox", { name: /语言/ }));
    await userEvent.click(await screen.findByRole("option", { name: nativeName("en") }));

    expect(localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe("en");
    expect(mockReload).toHaveBeenCalledTimes(1);
    expect(mockMutate).not.toHaveBeenCalled();
  });

  it("preserves the SSO setting and sign-in route in Chinese", async () => {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, "zh-CN");
    vi.mocked(useUIConfig).mockReturnValue({
      data: { sso_configured: true, auto_redirect_to_sso: false },
      isLoading: false,
    } as ReturnType<typeof useUIConfig>);
    renderLoginPage();

    expect(await screen.findByText("AUTO_REDIRECT_UI_LOGIN_TO_SSO=true")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "使用单点登录" }));
    expect(mockPush).toHaveBeenCalledWith(expect.stringContaining("http://localhost:4000/sso/key/generate?return_to="));
    await userEvent.click(screen.getByRole("button", { name: "关闭" }));
    expect(screen.queryByText("AUTO_REDIRECT_UI_LOGIN_TO_SSO=true")).not.toBeInTheDocument();
  });
});
