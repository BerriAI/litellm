import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Plugin } from "./types";

import SkillDetail from "./skill_detail";

const buildSkill = (source: Plugin["source"]): Plugin => ({
  id: "plugin-id",
  name: "my-skill",
  source,
  enabled: true,
});

describe("SkillDetail source", () => {
  it("links a github source to the repository", () => {
    render(<SkillDetail skill={buildSkill({ source: "github", repo: "org/repo" })} onBack={vi.fn()} />);
    expect(screen.getByRole("link", { name: /github.com\/org\/repo/ })).toHaveAttribute(
      "href",
      "https://github.com/org/repo",
    );
  });

  it("renders an ssh clone url as plain text instead of an unusable link", () => {
    render(
      <SkillDetail skill={buildSkill({ source: "url", url: "git@ghe.example.com:org/repo.git" })} onBack={vi.fn()} />,
    );
    expect(screen.getByText("git@ghe.example.com:org/repo.git")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /ghe.example.com/ })).not.toBeInTheDocument();
  });

  it("renders an ssh git-subdir source as plain text without a tree path", () => {
    render(
      <SkillDetail
        skill={buildSkill({ source: "git-subdir", url: "git@ghe.example.com:org/repo.git", path: "plugins/x" })}
        onBack={vi.fn()}
      />,
    );
    expect(screen.getByText("git@ghe.example.com:org/repo.git @ plugins/x")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /ghe.example.com/ })).not.toBeInTheDocument();
  });
});
