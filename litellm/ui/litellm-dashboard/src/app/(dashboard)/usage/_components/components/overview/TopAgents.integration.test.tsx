import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import TopAgents from "./TopAgents";
import type { TagSummaryRow } from "./agentCatalog";
import type { TrendSelection } from "./trendSelection";

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
const october = { start: "2026-10-01", end: "2026-10-31" };
const tokensDaily: TrendSelection = { metric: "tokens", granularity: "day" };
const shared = { daily: [], loading: false, range: october, selection: tokensDaily };

const metricsOf = (tokens: number, spend: number): SpendMetrics => ({
  spend,
  prompt_tokens: tokens,
  completion_tokens: 0,
  total_tokens: tokens,
  api_requests: 1,
  successful_requests: 1,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const tagDay = (date: string, tags: Record<string, number>): DailyData => {
  const total = Object.values(tags).reduce((sum, tokens) => sum + tokens, 0);
  return {
    date,
    metrics: metricsOf(total, 0),
    breakdown: {
      models: {},
      model_groups: {},
      mcp_servers: {},
      providers: {},
      api_keys: {},
      endpoints: {},
      entities: Object.fromEntries(
        Object.entries(tags).map(([tag, tokens]) => [
          tag,
          { metrics: metricsOf(tokens, 0), metadata: {}, api_key_breakdown: {} },
        ]),
      ),
    },
  };
};

describe("TopAgents row navigation", () => {
  it("should render recognizable client logos and open usage when the row totals are clicked", async () => {
    const user = userEvent.setup();
    const onOpenAgent = vi.fn();
    render(<TopAgents {...shared} rows={rows} totalTokens={600} onOpenAgent={onOpenAgent} />);

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
    render(<TopAgents {...shared} rows={rows} totalTokens={600} onOpenAgent={onOpenAgent} />);
    act(() => screen.getByRole("button", { name: "View Claude Desktop usage" }).focus());

    await user.keyboard("{Enter}");
    expect(onOpenAgent).toHaveBeenCalledWith(expect.objectContaining({ id: "claude-desktop" }));
  });

  it("should leave an unfilterable residual descriptive instead of making it an inaccurate drill", () => {
    const rollup = { ...row("User-Agent: claude-cli", 1000), total_requests: 2 };
    render(<TopAgents {...shared} rows={[rollup, rows[0]]} totalTokens={1000} onOpenAgent={vi.fn()} />);

    expect(screen.getByText("Claude (unspecified)")).toBeInTheDocument();
    expect(screen.getByText("Client not identified")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "View Claude (unspecified) usage" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "View Claude Desktop usage" })).toBeInTheDocument();
  });

  it("adds each day's attributed clients into months without counting a bare and a versioned tag twice", () => {
    const bare = "User-Agent: claude-cli";
    const daily = [
      tagDay("2026-10-05", { [bare]: 1337, [desktopTag]: 400 }),
      tagDay("2026-10-06", { [bare]: 1337, [desktopTag]: 400 }),
      tagDay("2026-11-05", { [bare]: 1337, [desktopTag]: 400 }),
    ];
    const rollup = { ...row(bare, 4011), total_requests: 3 };
    const selection: TrendSelection = { metric: "tokens", granularity: "month" };
    render(
      <TopAgents
        rows={[rollup, row(desktopTag, 1200)]}
        daily={daily}
        loading={false}
        totalTokens={4011}
        range={{ start: "2026-10-01", end: "2026-11-30" }}
        selection={selection}
      />,
    );

    expect(screen.getByText("Monthly tokens by client, recognised from each request's User-Agent tag")).toBeVisible();
    expect(screen.getByText("2.7K")).toBeInTheDocument();
    expect(screen.getByText("1.3K")).toBeInTheDocument();
    expect(screen.queryByText("3.5K")).not.toBeInTheDocument();
  });

  it("ranks and labels the rows by the metric the shared controls selected", () => {
    const tokensHeavy = { ...row(desktopTag, 900), total_spend: 1 };
    const spendHeavy = { ...row("User-Agent: claude-code/2.1.263", 100), total_spend: 40 };
    const spend: TrendSelection = { metric: "spend", granularity: "day" };
    const { rerender } = render(
      <TopAgents {...shared} rows={[tokensHeavy, spendHeavy]} totalTokens={1000} onOpenAgent={vi.fn()} />,
    );
    const order = () => screen.getAllByRole("button", { name: /^View .* usage$/ }).map((button) => button.textContent);

    expect(order()[0]).toContain("Claude Desktop");
    expect(order()[0]).toContain("900 tokens");

    rerender(
      <TopAgents
        {...shared}
        selection={spend}
        rows={[tokensHeavy, spendHeavy]}
        totalTokens={1000}
        onOpenAgent={vi.fn()}
      />,
    );

    expect(order()[0]).toContain("Claude Code");
    expect(order()[0]).toContain("$40");
    expect(screen.getByText("Daily spend by client, recognised from each request's User-Agent tag")).toBeVisible();
  });
});
