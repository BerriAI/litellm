import { act, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useState } from "react";
import { AuthProvider, useAuth } from "./AuthContext";
import DashboardLanguageProvider from "./DashboardLanguageProvider";
import { useLanguage } from "./LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";

const network = vi.hoisted(() => ({
  getUiConfig: vi.fn(),
  getUiSettings: vi.fn(),
  getProxyBaseUrl: vi.fn(() => "/proxy-ready"),
  setGlobalLitellmHeaderName: vi.fn(),
}));

vi.mock("@/components/networking", () => network);

function Content() {
  const { t } = useLanguage();
  const { setUserID } = useAuth();
  const [draft, setDraft] = useState("");
  return (
    <>
      <h1>{t("Home")}</h1>
      <LanguageSelector />
      <input aria-label="draft" value={draft} onChange={(event) => setDraft(event.target.value)} />
      <button onClick={() => setUserID("alice")}>alice</button>
      <button onClick={() => setUserID(null)}>guest</button>
    </>
  );
}

const tree = () => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}>
    <AuthProvider>
      <DashboardLanguageProvider>
        <Content />
      </DashboardLanguageProvider>
    </AuthProvider>
  </QueryClientProvider>
);

describe("应用根部语言上下文", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    network.getUiConfig.mockResolvedValue({});
    network.getUiSettings.mockResolvedValue({ values: { default_ui_language: "zh-CN" } });
  });

  it("等待运行时代理配置再读取全站语言，切换时不重新请求或丢失草稿", async () => {
    let ready = () => {};
    network.getUiConfig.mockReturnValue(
      new Promise<void>((resolve) => {
        ready = resolve;
      }),
    );
    render(tree());
    expect(network.getUiSettings).not.toHaveBeenCalled();
    await act(async () => ready());
    expect(await screen.findByRole("heading", { name: "首页" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: "draft" }), { target: { value: "Home" } });
    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(screen.getByRole("heading", { name: "Home" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "draft" })).toHaveValue("Home");
    expect(network.getUiSettings).toHaveBeenCalledTimes(1);
    expect(network.getUiConfig).toHaveBeenCalledTimes(1);
    expect(document.documentElement).toHaveAttribute("lang", "en");
  });

  it("登录后使用个人偏好，访客偏好不会覆盖账户选择", async () => {
    render(tree());
    await screen.findByRole("heading", { name: "首页" });
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "en" } });
    fireEvent.click(screen.getByRole("button", { name: "alice" }));
    expect(screen.getByRole("heading", { name: "首页" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "guest" }));
    expect(screen.getByRole("heading", { name: "Home" })).toBeInTheDocument();
    expect(network.getUiSettings).toHaveBeenCalledTimes(1);
  });
});
