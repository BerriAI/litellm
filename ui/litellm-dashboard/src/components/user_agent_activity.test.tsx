import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import UserAgentActivity from "./user_agent_activity";
import * as networking from "./networking";

// Mock the networking module
vi.mock("./networking", () => ({
  userAgentSummaryCall: vi.fn(),
  tagDauCall: vi.fn(),
  tagWauCall: vi.fn(),
  tagMauCall: vi.fn(),
  tagDistinctCall: vi.fn(),
}));

// Mock PerUserUsage component
vi.mock("./per_user_usage", () => ({
  default: () => <div>Per User Usage</div>,
}));

describe("UserAgentActivity", () => {
  const mockUserAgentSummaryCall = vi.mocked(networking.userAgentSummaryCall);
  const mockTagDauCall = vi.mocked(networking.tagDauCall);
  const mockTagWauCall = vi.mocked(networking.tagWauCall);
  const mockTagMauCall = vi.mocked(networking.tagMauCall);
  const mockTagDistinctCall = vi.mocked(networking.tagDistinctCall);

  const mockDistinctTagsData = {
    results: [{ tag: "User-Agent: Chrome/1.0" }, { tag: "User-Agent: Firefox/2.0" }, { tag: "User-Agent: Safari/3.0" }],
  };

  const mockSummaryData = {
    results: [
      {
        tag: "User-Agent: Chrome/1.0",
        unique_users: 100,
        total_requests: 1000,
        successful_requests: 950,
        failed_requests: 50,
        total_tokens: 50000,
        total_spend: 25.5,
      },
      {
        tag: "User-Agent: Firefox/2.0",
        unique_users: 80,
        total_requests: 800,
        successful_requests: 760,
        failed_requests: 40,
        total_tokens: 40000,
        total_spend: 20.3,
      },
    ],
  };

  const mockDauData = {
    results: [
      {
        tag: "User-Agent: Chrome/1.0",
        active_users: 50,
        date: "2025-01-01",
      },
      {
        tag: "User-Agent: Firefox/2.0",
        active_users: 30,
        date: "2025-01-01",
      },
    ],
  };

  const mockWauData = {
    results: [
      {
        tag: "User-Agent: Chrome/1.0",
        active_users: 200,
        date: "Week 1 (Jan 1)",
      },
    ],
  };

  const mockMauData = {
    results: [
      {
        tag: "User-Agent: Chrome/1.0",
        active_users: 500,
        date: "Month 1 (Jan)",
      },
    ],
  };

  const defaultProps = {
    accessToken: "test-token",
    userRole: "Admin",
    dateValue: {
      from: new Date("2025-01-01"),
      to: new Date("2025-01-31"),
    },
  };

  beforeEach(() => {
    mockUserAgentSummaryCall.mockClear();
    mockTagDauCall.mockClear();
    mockTagWauCall.mockClear();
    mockTagMauCall.mockClear();
    mockTagDistinctCall.mockClear();

    mockTagDistinctCall.mockResolvedValue(mockDistinctTagsData);
    mockUserAgentSummaryCall.mockResolvedValue(mockSummaryData);
    mockTagDauCall.mockResolvedValue(mockDauData);
    mockTagWauCall.mockResolvedValue(mockWauData);
    mockTagMauCall.mockResolvedValue(mockMauData);
  });

  it("should render summary cards with user agent data", async () => {
    render(<UserAgentActivity {...defaultProps} />);

    // Wait for data to load
    await waitFor(() => {
      expect(mockUserAgentSummaryCall).toHaveBeenCalled();
      expect(mockTagDistinctCall).toHaveBeenCalled();
    });

    // Check that summary section is displayed
    expect(screen.getByText("Summary by User Agent")).toBeInTheDocument();
    expect(screen.getByText("Performance metrics for different user agents")).toBeInTheDocument();

    // Check that user agent cards are displayed
    await waitFor(() => {
      expect(screen.getAllByText("Chrome/1.0").length).toBeGreaterThan(0);
      expect(screen.getAllByText("Firefox/2.0").length).toBeGreaterThan(0);
    });

    // Check that metrics are displayed
    expect(screen.getAllByText("Success Requests").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Total Tokens").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Total Cost").length).toBeGreaterThan(0);
  });

  it("should switch between DAU, WAU, and MAU tabs", async () => {
    render(<UserAgentActivity {...defaultProps} />);

    // Wait for data to load
    await waitFor(() => {
      expect(mockTagDauCall).toHaveBeenCalled();
      expect(mockTagWauCall).toHaveBeenCalled();
      expect(mockTagMauCall).toHaveBeenCalled();
    });

    // Check default DAU tab content
    expect(screen.getByText("Daily Active Users - Last 7 Days")).toBeInTheDocument();

    // Find all WAU tab buttons (there might be multiple)
    const wauTabs = screen.getAllByText("WAU");
    fireEvent.click(wauTabs[0]);

    // Check WAU tab content
    await waitFor(() => {
      expect(screen.getByText("Weekly Active Users - Last 7 Weeks")).toBeInTheDocument();
    });

    // Find all MAU tab buttons
    const mauTabs = screen.getAllByText("MAU");
    fireEvent.click(mauTabs[0]);

    // Check MAU tab content
    await waitFor(() => {
      expect(screen.getByText("Monthly Active Users - Last 7 Months")).toBeInTheDocument();
    });
  });

  it("should display filter dropdown and allow tag selection", async () => {
    render(<UserAgentActivity {...defaultProps} />);

    // Wait for tags to load
    await waitFor(() => {
      expect(mockTagDistinctCall).toHaveBeenCalled();
    });

    // Check that filter label is present
    expect(screen.getByText("Filter by User Agents")).toBeInTheDocument();

    // One library paints the prompt as its own text node and the other leaves it on the input's
    // placeholder attribute, so either one means the user is being told what the filter does.
    const prompts =
      screen.queryAllByText("All User Agents").length + screen.queryAllByPlaceholderText("All User Agents").length;
    expect(prompts).toBeGreaterThan(0);
  });

  // Walks up from the panel's heading to the nearest ancestor that owns a chart, so the
  // assertions do not depend on how many wrappers the tab library puts around a panel.
  const chartForTitle = (title: string): HTMLElement => {
    let node: HTMLElement | null = screen.getByText(title);
    while (node && !node.querySelector('[data-slot="chart"]')) {
      node = node.parentElement;
    }
    const chart = node?.querySelector('[data-slot="chart"]') ?? null;
    expect(chart).not.toBeNull();
    return chart as HTMLElement;
  };

  const expectStackedTwoCategoryChart = (chart: HTMLElement, firstBucketLabel: string) => {
    expect(chart.querySelectorAll(".recharts-bar")).toHaveLength(2);

    const rectangles = Array.from(chart.querySelectorAll("path.recharts-rectangle"));
    const fills = new Set(rectangles.map((rect) => rect.getAttribute("fill")));
    expect(fills).toEqual(new Set(["var(--color-blue-500, #3b82f6)", "var(--color-cyan-500, #06b6d4)"]));

    const xPositions = new Set(rectangles.map((rect) => rect.getAttribute("d")?.match(/^M\s*([\d.]+)/)?.[1]));
    expect(xPositions.size).toBe(1);

    expect(chart).toHaveTextContent(/Chrome\/1\.0/);
    expect(chart).toHaveTextContent(/Firefox\/2\.0/);
    expect(chart).toHaveTextContent(new RegExp(firstBucketLabel));

    const tickTexts = Array.from(chart.querySelectorAll(".recharts-cartesian-axis-tick-value")).map(
      (tick) => tick.textContent ?? "",
    );
    expect(tickTexts.some((tick) => /^\d+K$/.test(tick))).toBe(true);
  };

  it("keeps every tab panel mounted so switching tabs does not reset their state", async () => {
    render(<UserAgentActivity {...defaultProps} />);

    await waitFor(() => {
      expect(mockTagDauCall).toHaveBeenCalled();
    });

    // No tab has been clicked: the inactive DAU/WAU/MAU panels are mounted alongside the active one.
    expect(screen.getByText("Daily Active Users - Last 7 Days")).toBeInTheDocument();
    expect(screen.getByText("Weekly Active Users - Last 7 Weeks")).toBeInTheDocument();
    expect(screen.getByText("Monthly Active Users - Last 7 Months")).toBeInTheDocument();

    // And so is the second panel of the outer tab group.
    expect(screen.getByText("Per User Usage")).toBeInTheDocument();
  });

  it("renders the DAU chart stacked with default color cycle and abbreviated axis ticks", async () => {
    const firstBucketDate = new Date();
    firstBucketDate.setDate(firstBucketDate.getDate() - 6);
    const todayStr = new Date().toISOString().split("T")[0];
    mockTagDauCall.mockResolvedValue({
      results: [
        { tag: "User-Agent: Chrome/1.0", active_users: 4000, date: todayStr },
        { tag: "User-Agent: Firefox/2.0", active_users: 2600, date: todayStr },
      ],
    });

    render(<UserAgentActivity {...defaultProps} />);

    await waitFor(() => {
      expect(
        chartForTitle("Daily Active Users - Last 7 Days").querySelectorAll("path.recharts-rectangle"),
      ).toHaveLength(2);
    });

    expectStackedTwoCategoryChart(
      chartForTitle("Daily Active Users - Last 7 Days"),
      firstBucketDate.toISOString().split("T")[0],
    );
  });

  it("renders the WAU chart stacked with week buckets and abbreviated axis ticks", async () => {
    mockTagWauCall.mockResolvedValue({
      results: [
        { tag: "User-Agent: Chrome/1.0", active_users: 2000, date: "Week 3 (Jan 15)" },
        { tag: "User-Agent: Firefox/2.0", active_users: 1500, date: "Week 3 (Jan 15)" },
      ],
    });

    render(<UserAgentActivity {...defaultProps} />);

    await waitFor(() => {
      expect(
        chartForTitle("Weekly Active Users - Last 7 Weeks").querySelectorAll("path.recharts-rectangle"),
      ).toHaveLength(2);
    });

    expectStackedTwoCategoryChart(chartForTitle("Weekly Active Users - Last 7 Weeks"), "Week 1");
  });

  it("renders the MAU chart stacked with month buckets and abbreviated axis ticks", async () => {
    mockTagMauCall.mockResolvedValue({
      results: [
        { tag: "User-Agent: Chrome/1.0", active_users: 5000, date: "Month 2 (Feb)" },
        { tag: "User-Agent: Firefox/2.0", active_users: 3000, date: "Month 2 (Feb)" },
      ],
    });

    render(<UserAgentActivity {...defaultProps} />);

    await waitFor(() => {
      expect(
        chartForTitle("Monthly Active Users - Last 7 Months").querySelectorAll("path.recharts-rectangle"),
      ).toHaveLength(2);
    });

    expectStackedTwoCategoryChart(chartForTitle("Monthly Active Users - Last 7 Months"), "Month 1");
  });
});
