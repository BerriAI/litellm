import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { TooltipProvider } from "@/components/ui/tooltip";
import { toast } from "@/lib/toast";
import { CopyIconButton } from "./ModelName";

function CopyPreview() {
  return (
    <LanguageProvider defaultLanguage="zh-CN" userId="copy-language" storageScope="test">
      <LanguageSelector />
      <TooltipProvider>
        <CopyIconButton value="Home" label="Home" />
      </TooltipProvider>
    </LanguageProvider>
  );
}

describe("复制按钮语言", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  afterEach(() => vi.restoreAllMocks());

  it("复制原始值，只翻译提示，并支持切回英文", async () => {
    // userEvent 提供隔离剪贴板，避免访问系统剪贴板或真实凭据。
    const user = userEvent.setup();
    const writeText = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue(undefined);
    render(<CopyPreview />);
    await user.click(screen.getByRole("button", { name: "复制Home" }));
    expect(writeText).toHaveBeenCalledWith("Home");
    expect(toast.success).toHaveBeenLastCalledWith("已复制Home");
    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    await user.click(screen.getByRole("button", { name: "Copy Home" }));
    expect(writeText).toHaveBeenLastCalledWith("Home");
    expect(toast.success).toHaveBeenLastCalledWith("Home copied");
    expect(screen.queryByText("首页")).not.toBeInTheDocument();
  });

  it("剪贴板拒绝时显示中文失败提示", async () => {
    const user = userEvent.setup();
    vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(new Error("clipboard denied"));
    render(<CopyPreview />);
    await user.click(screen.getByRole("button", { name: "复制Home" }));
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith("无法复制到剪贴板"));
    expect(toast.success).not.toHaveBeenCalled();
  });
});
