import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import UsageBreakdownChart from "./UsageBreakdownChart";

const metrics: SpendMetrics = {
  spend: 9,
  total_tokens: 800,
  prompt_tokens: 600,
  completion_tokens: 200,
  api_requests: 3,
  successful_requests: 3,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
};
const results: DailyData[] = [
  {
    date: "2026-10-01",
    metrics,
    breakdown: {
      model_groups: { "claude-sonnet-4": { metrics, metadata: {}, api_key_breakdown: {} } },
      models: {},
      mcp_servers: {},
      providers: {},
      api_keys: {},
      entities: {},
    },
  },
];

describe("UsageBreakdownChart", () => {
  it("should render the current Usage chart controls and model leaderboard", () => {
    render(<UsageBreakdownChart results={results} loading={false} />);

    expect(screen.getByRole("heading", { name: "Daily usage" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Top models" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Spend" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Daily" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Linear" })).toBeChecked();
    expect(screen.getByText("claude-sonnet-4")).toBeInTheDocument();
    expect(within(screen.getByRole("listitem")).getByText("$9.00")).toBeInTheDocument();
  });

  it("should keep metric, time bucket and scale controls working in the compact panel", async () => {
    const user = userEvent.setup();
    const deploymentResults = results.map((day) => ({
      ...day,
      breakdown: { ...day.breakdown, model_groups: {}, models: day.breakdown.model_groups },
    }));
    render(<UsageBreakdownChart results={deploymentResults} loading={false} compact dimension="models" />);

    expect(screen.getByText("claude-sonnet-4")).toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: "Tokens" }));
    expect(screen.getByText("Daily tokens by model (top 8, rest grouped as Other)")).toBeInTheDocument();
    expect(within(screen.getByRole("listitem")).getByText("800")).toBeInTheDocument();
    expect(screen.queryByText("$9.00")).not.toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: "Weekly" }));
    expect(screen.getByRole("heading", { name: "Weekly usage" })).toBeInTheDocument();
    expect(screen.getByText("Weekly tokens by model (top 8, rest grouped as Other)")).toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: "Log" }));
    expect(screen.getByRole("radio", { name: "Log" })).toBeChecked();

    await user.click(screen.getByRole("radio", { name: "Requests" }));
    expect(screen.getByText("Weekly requests by model (top 8, rest grouped as Other)")).toBeInTheDocument();
    expect(within(screen.getByRole("listitem")).getByText("3")).toBeInTheDocument();
    expect(screen.queryByText("800")).not.toBeInTheDocument();
  });

  it("should show loading rather than an empty leaderboard until the data has settled", () => {
    render(<UsageBreakdownChart results={[]} loading compact />);

    expect(screen.getByRole("status", { name: "Loading top models" })).toBeInTheDocument();
    expect(screen.queryByText("No usage in this range")).not.toBeInTheDocument();
  });
});
