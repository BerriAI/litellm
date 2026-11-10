import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import TopAgents from "./TopAgents";
import type { TagSummaryRow } from "./agentCatalog";

const desktopTag = "User-Agent: claude-cli/2.1.263 (external, claude-desktop-3p)";
const row = (tag: string, tokens: number): TagSummaryRow => ({
  tag,
  total_spend: tokens / 100,
  total_tokens: tokens,
  total_requests: 1,
  successful_requests: 1,
  failed_requests: 0,
  unique_users: 1,
});
const rows = [row(desktopTag, 400), row("User-Agent: claude-code/2.1.263", 200)];

describe("TopAgents row navigation", () => {
  it("should render recognizable client logos and open usage when the row totals are clicked", async () => {
    const user = userEvent.setup();
    const onOpenAgent = vi.fn();
    render(<TopAgents rows={rows} daily={[]} loading={false} totalTokens={600} onOpenAgent={onOpenAgent} />);

    expect(screen.getByRole("heading", { name: "Top agents" })).toBeInTheDocument();
    const desktop = screen.getByRole("button", { name: "View Claude Desktop usage" });
    expect(within(desktop).getByRole("img", { name: "Claude Desktop logo" })).toHaveAttribute(
      "src",
      expect.any(String),
    );
    expect(screen.getByRole("img", { name: "Claude Code logo" })).toBeInTheDocument();

    await user.click(within(desktop).getByText("400 tokens"));
    expect(onOpenAgent).toHaveBeenCalledOnce();
    expect(onOpenAgent).toHaveBeenCalledWith(expect.objectContaining({ id: "claude-desktop", tags: [desktopTag] }));
  });

  it("should open a focused agent row with the keyboard", async () => {
    const user = userEvent.setup();
    const onOpenAgent = vi.fn();
    render(<TopAgents rows={rows} daily={[]} loading={false} totalTokens={600} onOpenAgent={onOpenAgent} />);
    act(() => screen.getByRole("button", { name: "View Claude Desktop usage" }).focus());

    await user.keyboard("{Enter}");
    expect(onOpenAgent).toHaveBeenCalledWith(expect.objectContaining({ id: "claude-desktop" }));
  });

  it("should leave an unfilterable residual descriptive instead of making it an inaccurate drill", () => {
    const rollup = { ...row("User-Agent: claude-cli", 1000), total_requests: 2 };
    render(<TopAgents rows={[rollup, rows[0]]} daily={[]} loading={false} totalTokens={1000} onOpenAgent={vi.fn()} />);

    expect(screen.getByText("Claude (unspecified)")).toBeInTheDocument();
    expect(screen.getByText("Client not identified")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "View Claude (unspecified) usage" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "View Claude Desktop usage" })).toBeInTheDocument();
  });
});
