import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";

import SkillDetail from "./skill_detail";
import type { Plugin } from "./types";

const skill: Plugin = {
  id: "plugin-1",
  name: "my-skill",
  source: { source: "github", repo: "acme/my-skill" },
  enabled: true,
};

const buildSkill = (source: Plugin["source"]): Plugin => ({
  id: "plugin-id",
  name: "my-skill",
  source,
  enabled: true,
});

const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0];

describe("SkillDetail tab URL state", () => {
  it("opens on Overview when the URL names no tab", () => {
    renderWithProviders(<SkillDetail skill={skill} onBack={vi.fn()} />);
    expect(screen.getByText("Skill Details")).toBeInTheDocument();
    expect(screen.queryByText("Using this skill")).not.toBeInTheDocument();
  });

  it.each([
    { tab: "usage", heading: "Using this skill" },
    { tab: "setup", heading: "One-time marketplace setup" },
  ])("opens the $tab tab named by skill_tab", ({ tab, heading }) => {
    renderWithProviders(<SkillDetail skill={skill} onBack={vi.fn()} />, {
      searchParams: `?skill=plugin-1&skill_tab=${tab}`,
    });
    expect(screen.getByText(heading)).toBeInTheDocument();
    expect(screen.queryByText("Skill Details")).not.toBeInTheDocument();
  });

  it("writes skill_tab as the user moves between tabs and drops it on Overview", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<SkillDetail skill={skill} onBack={vi.fn()} />, {
      searchParams: "?skill=plugin-1",
      onUrlUpdate,
    });

    await user.click(screen.getByText("How to Use"));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("skill_tab")).toBe("usage"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("skill")).toBe("plugin-1");

    await user.click(screen.getByText(/See one-time setup/));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("skill_tab")).toBe("setup"));
    expect(screen.getByText("One-time marketplace setup")).toBeInTheDocument();

    await user.click(screen.getByText("Overview"));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("skill_tab")).toBe(false));
    expect(screen.getByText("Skill Details")).toBeInTheDocument();
  });

  it("falls back to Overview for an unknown skill_tab and clears it from the URL", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    render(<SkillDetail skill={skill} onBack={vi.fn()} />, {
      wrapper: ({ children }: { children: ReactNode }) => (
        <NuqsTestingAdapter
          searchParams="?skill=plugin-1&skill_tab=bogus"
          onUrlUpdate={onUrlUpdate}
          hasMemory
          resetUrlUpdateQueueOnMount={false}
        >
          {children}
        </NuqsTestingAdapter>
      ),
    });

    expect(screen.getByText("Skill Details")).toBeInTheDocument();
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("skill_tab")).toBe(false));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("skill")).toBe("plugin-1");
  });

  it("clears skill_tab and calls onBack when leaving the detail", async () => {
    const user = userEvent.setup();
    const onBack = vi.fn();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<SkillDetail skill={skill} onBack={onBack} />, {
      searchParams: "?skill=plugin-1&skill_tab=usage",
      onUrlUpdate,
    });

    await user.click(screen.getByText("Skills"));

    expect(onBack).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("skill_tab")).toBe(false));
  });
});

describe("SkillDetail source", () => {
  it("links a github source to the repository", () => {
    renderWithProviders(<SkillDetail skill={buildSkill({ source: "github", repo: "org/repo" })} onBack={vi.fn()} />);
    expect(screen.getByRole("link", { name: "github.com/org/repo" })).toHaveAttribute(
      "href",
      "https://github.com/org/repo",
    );
  });

  it("renders an ssh clone url as plain text instead of an unusable link", () => {
    renderWithProviders(
      <SkillDetail skill={buildSkill({ source: "url", url: "git@ghe.example.com:org/repo.git" })} onBack={vi.fn()} />,
    );
    expect(screen.getByText("git@ghe.example.com:org/repo.git")).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("renders an ssh git-subdir source as plain text without a tree path", () => {
    renderWithProviders(
      <SkillDetail
        skill={buildSkill({ source: "git-subdir", url: "git@ghe.example.com:org/repo.git", path: "plugins/x" })}
        onBack={vi.fn()}
      />,
    );
    expect(screen.getByText("git@ghe.example.com:org/repo.git @ plugins/x")).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });
});
