import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LanguageProvider, useLanguage } from "./LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";

function Preview() {
  const { t } = useLanguage();
  return (
    <>
      <h1>{t("Home")}</h1>
      <LanguageSelector />
    </>
  );
}

const tree = (userId = "alice", defaultLanguage = "zh-CN", storageScope = "/proxy-a") => (
  <LanguageProvider userId={userId} defaultLanguage={defaultLanguage} storageScope={storageScope}>
    <Preview />
  </LanguageProvider>
);

describe("语言偏好链路", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("跟随默认、立即切换、刷新记忆，并允许恢复跟随", () => {
    const view = render(tree());
    expect(screen.getByRole("heading", { name: "首页" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(screen.getByRole("heading", { name: "Home" })).toBeInTheDocument();
    expect(document.documentElement).toHaveAttribute("lang", "en");
    view.unmount();
    render(tree());
    expect(screen.getByRole("heading", { name: "Home" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "default" } });
    expect(screen.getByRole("heading", { name: "首页" })).toBeInTheDocument();
    expect(document.documentElement).toHaveAttribute("lang", "zh-CN");
  });

  it("隔离用户和代理的偏好，跟随者响应全站默认变更", () => {
    const view = render(tree());
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "en" } });
    view.rerender(tree("bob"));
    expect(screen.getByRole("heading", { name: "首页" })).toBeInTheDocument();
    view.rerender(tree("bob", "en"));
    expect(screen.getByRole("heading", { name: "Home" })).toBeInTheDocument();
    view.rerender(tree("alice"));
    expect(screen.getByRole("heading", { name: "Home" })).toBeInTheDocument();
    view.rerender(tree("alice", "zh-CN", "/proxy-b"));
    expect(screen.getByRole("heading", { name: "首页" })).toBeInTheDocument();
  });

  it("浏览器禁止存储时仍可切换，并说明选择无法保存", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("blocked");
    });
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    render(tree());
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "en" } });
    expect(screen.getByRole("heading", { name: "Home" })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(
      "Your language applies to this session only because browser storage is unavailable.",
    );
  });
});
