import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Plugin } from "./types";

import SkillDetail from "./skill_detail";

const { mockUseMarketplaceRequiresKey } = vi.hoisted(() => ({ mockUseMarketplaceRequiresKey: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/claudeCodeMarketplace/useClaudeCodeMarketplaceRequiresKey", () => ({
  useClaudeCodeMarketplaceRequiresKey: () => mockUseMarketplaceRequiresKey(),
}));

beforeEach(() => {
  mockUseMarketplaceRequiresKey.mockReturnValue(false);
});

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

describe("SkillDetail marketplace setup", () => {
  const renderSetupTab = () => {
    render(<SkillDetail skill={buildSkill({ source: "github", repo: "org/repo" })} onBack={vi.fn()} />);
    fireEvent.click(screen.getByText("How to Use"));
    fireEvent.click(screen.getByText(/See one-time setup/));
  };

  it("offers the add command and a keyless snippet when the marketplace is open", () => {
    renderSetupTab();
    expect(screen.getByText(/^\/plugin marketplace add .*\/claude-code\/marketplace\.json$/)).toBeInTheDocument();
    expect(screen.queryByText(/Bearer/)).not.toBeInTheDocument();
    expect(screen.queryByText(/\/plugin marketplace remove/)).not.toBeInTheDocument();
  });

  it("replaces the add command with a snippet carrying the key header when the marketplace requires a key", () => {
    mockUseMarketplaceRequiresKey.mockReturnValue(true);
    renderSetupTab();
    expect(screen.queryByText(/\/plugin marketplace add/)).not.toBeInTheDocument();
    expect(screen.getByText(/"Authorization": "Bearer <your-litellm-key>"/)).toBeInTheDocument();
    expect(screen.getByText("/plugin marketplace remove litellm")).toBeInTheDocument();
    expect(screen.getByText(/uninstalls the plugins you installed from it/)).toBeInTheDocument();
  });
});
