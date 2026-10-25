import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";

import { Plugin } from "./types";

import SkillDetail from "./skill_detail";

const buildSkill = (source: Plugin["source"]): Plugin => ({
  id: "plugin-id",
  name: "my-skill",
  source,
  enabled: true,
  version: null,
  description: null,
  created_at: null,
  updated_at: null,
});

describe("SkillDetail source", () => {
  it("翻译固定选项卡和属性名，保留同名用户数据", () => {
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="skill-detail" storageScope="skill-detail">
        <SkillDetail
          skill={{
            ...buildSkill({ source: "github", repo: "org/home" }),
            name: "Home",
            category: "Home",
            author: { name: "Settings" },
          }}
          onBack={vi.fn()}
        />
      </LanguageProvider>,
    );
    expect(screen.getByText("概览", { exact: true })).toBeVisible();
    expect(screen.getByText("使用方法", { exact: true })).toBeVisible();
    expect(screen.getByRole("cell", { name: "类别", exact: true })).toBeVisible();
    expect(screen.getByRole("cell", { name: "作者", exact: true })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Home", exact: true })).toBeVisible();
    expect(screen.getByRole("cell", { name: "Home", exact: true })).toBeVisible();
    expect(screen.getByRole("cell", { name: "Settings", exact: true })).toBeVisible();
  });

  it.each(["install", "settings"] as const)(
    "复制 %s 使用固定状态键，切语言后仍显示已复制且命令不变",
    async (target) => {
      const user = userEvent.setup();
      const clipboard = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue();
      render(
        <LanguageProvider defaultLanguage="en" userId={`skill-copy-${target}`} storageScope="skill-copy">
          <LanguageSelector />
          <SkillDetail skill={buildSkill({ source: "github", repo: "org/home" })} onBack={vi.fn()} />
        </LanguageProvider>,
      );
      fireEvent.click(screen.getByText("How to Use", { exact: true }));
      if (target === "settings") fireEvent.click(screen.getByText("See one-time setup →", { exact: true }));
      const command = target === "install" ? "/plugin install my-skill@litellm" : "~/.claude/settings.json";
      fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
      const copyButtons = screen.getAllByRole("button", { name: "复制", exact: true });
      await user.click(copyButtons[target === "install" ? 0 : 1]);
      expect(screen.getByRole("button", { name: "已复制", exact: true })).toBeVisible();
      fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
      expect(screen.getByRole("button", { name: "Copied", exact: true })).toBeVisible();
      const copied = clipboard.mock.calls[0][0];
      if (target === "install") expect(copied).toBe(command);
      else
        expect(JSON.parse(copied)).toEqual({
          extraKnownMarketplaces: {
            litellm: { source: { source: "url", url: `${window.location.origin}/claude-code/marketplace.json` } },
          },
        });
      expect(clipboard).toHaveBeenCalledOnce();
      clipboard.mockRestore();
    },
  );
  it("renders a skill whose API metadata is null", () => {
    render(
      <SkillDetail
        skill={{ ...buildSkill({ source: "github", repo: "org/repo" }), keywords: null, author: null, category: null }}
        onBack={vi.fn()}
      />,
    );
    expect(screen.getByRole("heading", { name: "my-skill" })).toBeInTheDocument();
    expect(screen.getByText("Public")).toBeInTheDocument();
  });

  it("links a github source to the repository", () => {
    render(<SkillDetail skill={buildSkill({ source: "github", repo: "org/repo" })} onBack={vi.fn()} />);
    expect(screen.getByRole("link", { name: "github.com/org/repo" })).toHaveAttribute(
      "href",
      "https://github.com/org/repo",
    );
  });

  it("renders an ssh clone url as plain text instead of an unusable link", () => {
    render(
      <SkillDetail skill={buildSkill({ source: "url", url: "git@ghe.example.com:org/repo.git" })} onBack={vi.fn()} />,
    );
    expect(screen.getByText("git@ghe.example.com:org/repo.git")).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("renders an ssh git-subdir source as plain text without a tree path", () => {
    render(
      <SkillDetail
        skill={buildSkill({ source: "git-subdir", url: "git@ghe.example.com:org/repo.git", path: "plugins/x" })}
        onBack={vi.fn()}
      />,
    );
    expect(screen.getByText("git@ghe.example.com:org/repo.git @ plugins/x")).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });
});
