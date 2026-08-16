import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import PerUserUsage from "./per_user_usage";
import * as networking from "./networking";

vi.mock("./networking", () => ({
  perUserAnalyticsCall: vi.fn(),
}));

type UserRow = {
  user_id: string;
  user_email: string | null;
  user_agent: string | null;
  successful_requests: number;
  failed_requests: number;
  total_requests: number;
  total_tokens: number;
  spend: number;
};

const userRow = (userId: string, userAgent: string | null, successfulRequests: number): UserRow => ({
  user_id: userId,
  user_email: null,
  user_agent: userAgent,
  successful_requests: successfulRequests,
  failed_requests: 0,
  total_requests: successfulRequests,
  total_tokens: 100,
  spend: 1,
});

describe("PerUserUsage", () => {
  const mockPerUserAnalyticsCall = vi.mocked(networking.perUserAnalyticsCall);

  const mockResponse = {
    results: [
      userRow("u1", "curl/8.0", 5),
      userRow("u2", "curl/8.0", 50),
      userRow("u3", "curl/8.0", 8),
      userRow("u4", null, 7),
      userRow("u5", null, 500),
    ],
    total_count: 5,
    page: 1,
    page_size: 50,
    total_pages: 1,
  };

  const defaultProps = {
    accessToken: "test-token",
    selectedTags: [],
    formatAbbreviatedNumber: (value: number) => String(value),
  };

  beforeEach(() => {
    mockPerUserAnalyticsCall.mockClear();
    mockPerUserAnalyticsCall.mockResolvedValue(mockResponse);
  });

  it("renders the user details table by default", async () => {
    render(<PerUserUsage {...defaultProps} />);

    await waitFor(() => {
      expect(mockPerUserAnalyticsCall).toHaveBeenCalled();
    });

    expect(screen.getByText("Per User Usage")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText("u1")).toBeInTheDocument();
    });
  });

  it("renders the usage distribution as a stacked bar chart with the explicit palette and users formatter", async () => {
    render(<PerUserUsage {...defaultProps} />);

    await waitFor(() => {
      expect(mockPerUserAnalyticsCall).toHaveBeenCalled();
    });

    fireEvent.click(screen.getByText("Usage Distribution"));

    const panel = screen.getByText("User Usage Distribution").closest("div")?.parentElement;
    expect(panel).not.toBeNull();

    await waitFor(() => {
      expect(panel!.querySelectorAll("path.recharts-rectangle")).toHaveLength(4);
    });

    const chart = panel!.querySelector('[data-slot="chart"]');
    expect(chart).not.toBeNull();
    expect(chart!.querySelectorAll(".recharts-bar")).toHaveLength(2);

    const rectangles = Array.from(chart!.querySelectorAll("path.recharts-rectangle"));
    const fills = new Set(rectangles.map((rect) => rect.getAttribute("fill")));
    expect(fills).toEqual(new Set(["var(--color-blue-500, #3b82f6)", "var(--color-green-500, #22c55e)"]));

    const xPositions = new Set(rectangles.map((rect) => rect.getAttribute("d")?.match(/^M\s*([\d.]+)/)?.[1]));
    expect(xPositions.size).toBe(3);

    expect(chart!.textContent).toContain("curl/8.0");
    expect(chart!.textContent).toContain("Unknown");
    for (const bucket of [
      "1-9 requests",
      "10-99 requests",
      "100-999 requests",
      "1K-9.9K requests",
      "10K-99.9K requests",
      "100K+ requests",
    ]) {
      expect(chart!.textContent).toContain(bucket);
    }

    const tickTexts = Array.from(chart!.querySelectorAll(".recharts-cartesian-axis-tick-value")).map(
      (tick) => tick.textContent ?? "",
    );
    expect(tickTexts.some((tick) => / users$/.test(tick))).toBe(true);
  });
});
