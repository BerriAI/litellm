import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { Toaster } from "@/components/ui/sonner";
import { ApiError } from "@/lib/http/client";
import { toast } from "@/lib/toast";

vi.unmock("@/lib/toast");

function LocalizedToaster() {
  return (
    <LanguageProvider defaultLanguage="zh-CN" userId="toast-language" storageScope="test">
      <Toaster expand />
      <LanguageSelector />
    </LanguageProvider>
  );
}

// 使用真实 Sonner 渲染标题，错误详情及普通通知文案均视为调用方内容。
describe("错误通知固定标题语言", () => {
  beforeEach(() => {
    localStorage.clear();
    toast.dismiss();
  });

  afterEach(() => act(() => toast.dismiss()));

  it("中文显示固定权限标题，切回英文时现有通知同步更新且不翻译详情", async () => {
    render(<LocalizedToaster />);
    act(() => toast.fromError(new ApiError("Home", 403, { error: { message: "Home" } })));
    expect(await screen.findByText("无访问权限")).toBeVisible();
    expect(screen.getByText("Home")).toBeVisible();
    expect(screen.queryByText("首页")).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(await screen.findByText("Access Denied")).toBeVisible();
    expect(screen.getByText("Home")).toBeVisible();
  });

  it("翻译无状态码的通用错误标题，保留错误原文", async () => {
    render(<LocalizedToaster />);
    act(() => toast.fromError(new Error("Home")));
    expect(await screen.findByText("错误")).toBeVisible();
    expect(screen.getByText("Home")).toBeVisible();
  });

  it("没有语言上下文时保留英文回退", async () => {
    render(<Toaster />);
    act(() => toast.fromError(new ApiError("Home", 429, "Home")));
    expect(await screen.findByText("Rate Limit Exceeded")).toBeVisible();
    expect(screen.getByText("Home")).toBeVisible();
  });

  it("普通成功和错误通知不隐式翻译传入内容", async () => {
    render(<LocalizedToaster />);
    act(() => {
      toast.success("Home");
      toast.error("Settings");
    });
    expect(await screen.findByText("Home")).toBeVisible();
    expect(screen.getByText("Settings")).toBeVisible();
    expect(screen.queryByText("首页")).not.toBeInTheDocument();
    expect(screen.queryByText("设置")).not.toBeInTheDocument();
  });
});
