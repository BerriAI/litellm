import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { copyToClipboard } from "@/utils/dataUtils";
import { LensAgentFix } from "./LensAgentFix";
import type { AgentFix } from "./lensData";

vi.mock("@/utils/dataUtils", () => ({ copyToClipboard: vi.fn() }));

const fix: AgentFix = {
  problem: "The agent has no repository tool, so it cannot push a branch.",
  user_goal: "Open a pull request fixing a typo",
  what_happened: "It replied that it lacked repository access",
  options: [
    { title: "Grant the repository tool", change: "Add the repository MCP server to the agent's tools" },
    { title: "Hand back a patch", change: "When pushing fails, return a patch the user can apply" },
  ],
  test_cases: [{ input: "Open a PR fixing the typo", expected: "A PR URL is returned" }],
};

describe("LensAgentFix", () => {
  it("shows the scoped problem and copies the chosen option as a complete prompt", async () => {
    const user = userEvent.setup();
    render(<LensAgentFix title="PRs were never opened" fix={fix} />);
    expect(screen.getAllByText(fix.problem)[0]).toBeVisible();
    expect(screen.getAllByText(fix.user_goal)[0]).toBeVisible();
    expect(screen.getAllByText(new RegExp(fix.test_cases[0].input))[0]).toBeVisible();
    expect(screen.getByText(fix.options[1].change)).not.toBeVisible();
    await user.click(screen.getByText(/Option 2: Hand back a patch/));
    expect(screen.getByText(fix.options[1].change)).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Copy option 2 prompt for Claude Code or Codex" }));
    const copied = vi.mocked(copyToClipboard).mock.calls[0][0] ?? "";
    expect(copied).toContain("# Fix: PRs were never opened");
    expect(copied).toContain(fix.options[1].change);
    expect(copied).not.toContain(fix.options[0].change);
    expect(copied).toContain(fix.test_cases[0].expected);
  });
});
