import { act, fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import LanguageSelector from "@/components/LanguageSelector";
import { Toaster } from "@/components/ui/sonner";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { ApiError } from "@/lib/http/client";
import { toast } from "@/lib/toast";
import { disableClaudeCodePlugin, enableClaudeCodePlugin } from "../networking";
import MakeSkillPublicForm from "./MakeSkillPublicForm";
import type { Plugin } from "./types";

vi.unmock("@/lib/toast");
vi.mock("../networking", () => ({
  enableClaudeCodePlugin: vi.fn(),
  disableClaudeCodePlugin: vi.fn().mockResolvedValue({ status: "success" }),
}));

// 同名界面词条作为业务数据，验证选择值不会被翻译。
const skills: Plugin[] = [
  {
    id: "skill-home",
    name: "Home",
    enabled: false,
    source: { source: "github", repo: "org/home" },
    description: null,
    created_at: null,
    updated_at: null,
    version: null,
  },
  {
    id: "skill-settings",
    name: "Settings",
    enabled: true,
    source: { source: "github", repo: "org/settings" },
    description: null,
    created_at: null,
    updated_at: null,
    version: null,
  },
];

describe("技能发布语言切换", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    toast.dismiss();
  });

  afterEach(() => act(() => toast.dismiss()));

  it.each(["success", "failure", "forbidden"] as const)(
    "%s 通知随当前语言显示，语言切换保留选择且不重复发布",
    async (outcome) => {
      const pending = Promise.withResolvers<Awaited<ReturnType<typeof enableClaudeCodePlugin>>>();
      vi.mocked(enableClaudeCodePlugin).mockReturnValueOnce(pending.promise);
      const onSuccess = vi.fn();
      const onClose = vi.fn();
      const user = userEvent.setup();
      renderWithProviders(
        <LanguageProvider defaultLanguage="en" userId={`publish-${outcome}`} storageScope="test">
          <LanguageSelector />
          <MakeSkillPublicForm
            visible
            accessToken="synthetic-token"
            skillsList={skills}
            onSuccess={onSuccess}
            onClose={onClose}
          />
          <Toaster expand />
        </LanguageProvider>,
      );

      await user.click(screen.getByRole("checkbox", { name: "Home", exact: true }));
      await user.click(screen.getByRole("checkbox", { name: "Settings", exact: true }));
      fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
      expect(screen.getByRole("checkbox", { name: "Home", exact: true })).toBeChecked();
      expect(screen.getByRole("checkbox", { name: "Settings", exact: true })).not.toBeChecked();
      expect(enableClaudeCodePlugin).not.toHaveBeenCalled();
      fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
      await user.click(screen.getByRole("button", { name: "Next", exact: true }));
      await user.click(screen.getByRole("button", { name: "Publish to Hub", exact: true }));
      expect(enableClaudeCodePlugin).toHaveBeenCalledExactlyOnceWith("synthetic-token", "Home");
      expect(disableClaudeCodePlugin).toHaveBeenCalledExactlyOnceWith("synthetic-token", "Settings");
      fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });

      await act(async () => {
        if (outcome === "success") pending.resolve({ status: "success" });
        else if (outcome === "forbidden") pending.reject(new ApiError("Home", 403, { error: { message: "Home" } }));
        else pending.reject(new Error("Home"));
      });

      const chinese = outcome === "success" ? "技能中心已更新，已发布 1 项技能" : "更新技能失败，请重试。";
      expect(await screen.findByText(chinese, { exact: true })).toBeVisible();
      if (outcome === "forbidden") expect(screen.getByText("无访问权限", { exact: true })).toBeVisible();
      fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
      const english =
        outcome === "success"
          ? "Skill Hub updated — 1 skill(s) published"
          : "Failed to update skills. Please try again.";
      expect(screen.getByText(english, { exact: true })).toBeVisible();
      if (outcome === "forbidden") expect(screen.getByText("Access Denied", { exact: true })).toBeVisible();
      expect(enableClaudeCodePlugin).toHaveBeenCalledOnce();
      expect(disableClaudeCodePlugin).toHaveBeenCalledOnce();
      if (outcome === "success") {
        expect(onSuccess).toHaveBeenCalledOnce();
        expect(onClose).toHaveBeenCalledOnce();
      } else {
        expect(onSuccess).not.toHaveBeenCalled();
        expect(onClose).not.toHaveBeenCalled();
        await user.click(screen.getByRole("button", { name: "Previous", exact: true }));
        expect(screen.getByRole("checkbox", { name: "Home", exact: true })).toBeChecked();
        expect(screen.getByRole("checkbox", { name: "Settings", exact: true })).not.toBeChecked();
      }
    },
  );
});
