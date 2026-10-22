import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { OverviewTotals } from "./overviewData";
import UsageStatStrip from "./UsageStatStrip";

const totals: OverviewTotals = {
  spend: 42.5,
  requests: 10,
  successful: 8,
  failed: 2,
  tokens: 9123456,
  inputTokens: 8123456,
  outputTokens: 1000000,
  cacheReadTokens: 5123456,
  cacheWriteTokens: 2000000,
  avgCostPerRequest: 4.25,
  avgLatencyMs: 250,
  successRate: 80,
};

const defaultProps = {
  results: [],
  totals,
  loading: false,
  requestCountsPending: false,
  budget: null,
};

describe("UsageStatStrip token breakdown", () => {
  it("should render the existing totals and open the full token counts", async () => {
    const user = userEvent.setup();
    render(<UsageStatStrip {...defaultProps} />);

    expect(screen.getByText("Total Tokens")).toBeInTheDocument();
    expect(screen.getByText("ok")).toBeInTheDocument();
    expect(screen.getByText("failed")).toBeInTheDocument();
    expect(screen.getByText("80.0% success rate")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "View token breakdown" }));
    const breakdown = await screen.findByRole("dialog", { name: "What counts toward total tokens?" });
    expect(breakdown).toHaveTextContent(/Input tokens\s*8,123,456/);
    expect(breakdown).toHaveTextContent(/Cache reads\s*5,123,456/);
    expect(breakdown).toHaveTextContent(/Cache writes\s*2,000,000/);
    expect(breakdown).toHaveTextContent(/Output tokens\s*1,000,000/);
    expect(breakdown).toHaveTextContent(/Total tokens\s*9,123,456/);
    expect(breakdown).toHaveTextContent("Input includes cache reads and writes.");
    expect(breakdown).toHaveTextContent("Compare the same person, client and time range");
  });

  it("should hide the token breakdown and request counts while usage is loading", () => {
    render(<UsageStatStrip {...defaultProps} loading requestCountsPending />);

    expect(screen.getByText("Total Tokens")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "View token breakdown" })).not.toBeInTheDocument();
    expect(screen.queryByText("ok")).not.toBeInTheDocument();
    expect(screen.queryByText("failed")).not.toBeInTheDocument();
  });
});
