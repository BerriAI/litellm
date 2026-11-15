import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

// The distribution panel owns the only chart in this component, so resolving it by slot keeps
// the assertions independent of how many wrappers the tab library puts around a panel.
const distributionChart = (): HTMLElement => {
  const chart = document.querySelector('[data-slot="chart"]');
  expect(chart).not.toBeNull();
  return chart as HTMLElement;
};

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

  it("keeps both tab panels mounted so switching tabs does not reset their state", async () => {
    render(<PerUserUsage {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("u1")).toBeInTheDocument();
    });

    // Still on the User Details tab: the distribution panel is mounted alongside it.
    expect(screen.getByText("User Usage Distribution")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Usage Distribution"));

    // And the details panel survives the switch rather than unmounting.
    expect(screen.getByText("u1")).toBeInTheDocument();
  });

  describe("server pagination", () => {
    const TOTAL_USERS = 120;

    const pageOfUsers = (page: number, pageSize: number, total: number): UserRow[] => {
      const start = (page - 1) * pageSize;
      const count = Math.max(0, Math.min(pageSize, total - start));
      return Array.from({ length: count }, (_, index) => userRow(`user-${start + index + 1}`, "curl/8.0", 5));
    };

    const serveUsers = (total: number) => {
      mockPerUserAnalyticsCall.mockImplementation(async (_token, page = 1, pageSize = 50) => ({
        results: pageOfUsers(page, pageSize, total),
        total_count: total,
        page,
        page_size: pageSize,
        total_pages: Math.ceil(total / pageSize),
      }));
    };

    beforeEach(() => {
      serveUsers(TOTAL_USERS);
    });

    const lastCall = () => mockPerUserAnalyticsCall.mock.calls[mockPerUserAnalyticsCall.mock.calls.length - 1];

    it("renders every row the server returns and shows the range from total_count", async () => {
      render(<PerUserUsage {...defaultProps} />);

      expect(await screen.findByText("user-50")).toBeInTheDocument();
      expect(screen.getByText("user-1")).toBeInTheDocument();
      expect(screen.getAllByRole("row")).toHaveLength(51);
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-50 of 120");
      expect(screen.getByTestId("pagination-prev")).toBeDisabled();
      expect(screen.getByTestId("pagination-next")).toBeEnabled();
    });

    it("refetches the next page when Next is clicked", async () => {
      const user = userEvent.setup();
      render(<PerUserUsage {...defaultProps} />);
      await screen.findByText("user-1");

      await user.click(screen.getByTestId("pagination-next"));

      expect(await screen.findByText("user-51")).toBeInTheDocument();
      expect(lastCall()).toEqual(["test-token", 2, 50, undefined]);
      expect(screen.queryByText("user-1")).not.toBeInTheDocument();
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 51-100 of 120");
      expect(screen.getByTestId("pagination-prev")).toBeEnabled();
    });

    it("disables Next once the response says this is the last page", async () => {
      const user = userEvent.setup();
      render(<PerUserUsage {...defaultProps} />);
      await screen.findByText("user-1");

      await user.click(screen.getByTestId("pagination-last"));

      expect(await screen.findByText("user-120")).toBeInTheDocument();
      expect(lastCall()).toEqual(["test-token", 3, 50, undefined]);
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 101-120 of 120");
      expect(screen.getByTestId("pagination-next")).toBeDisabled();
    });

    it("falls back to the last existing page when the data shrinks under the current page", async () => {
      const user = userEvent.setup();
      render(<PerUserUsage {...defaultProps} />);
      await screen.findByText("user-1");
      await user.click(screen.getByTestId("pagination-next"));
      await screen.findByText("user-51");

      serveUsers(60);
      await user.click(screen.getByTestId("pagination-next"));

      expect(await screen.findByText("user-60")).toBeInTheDocument();
      expect(mockPerUserAnalyticsCall.mock.calls.slice(-2)).toEqual([
        ["test-token", 3, 50, undefined],
        ["test-token", 2, 50, undefined],
      ]);
      expect(screen.getAllByRole("row")).toHaveLength(11);
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 51-60 of 60");
      expect(screen.getByTestId("pagination-next")).toBeDisabled();
    });

    it("goes back to the first page when the data disappears under the current page", async () => {
      const user = userEvent.setup();
      render(<PerUserUsage {...defaultProps} />);
      await screen.findByText("user-1");
      await user.click(screen.getByTestId("pagination-next"));
      await screen.findByText("user-51");

      serveUsers(0);
      await user.click(screen.getByTestId("pagination-next"));

      await waitFor(() => {
        expect(lastCall()).toEqual(["test-token", 1, 50, undefined]);
      });
      expect(mockPerUserAnalyticsCall.mock.calls.slice(-2)).toEqual([
        ["test-token", 3, 50, undefined],
        ["test-token", 1, 50, undefined],
      ]);
      expect(screen.getByText("No per-user usage data")).toBeInTheDocument();
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("No results");
      expect(screen.getByText("Page 1 of 1")).toBeInTheDocument();
      expect(screen.getByTestId("pagination-first")).toBeDisabled();
      expect(screen.getByTestId("pagination-prev")).toBeDisabled();
    });

    it("refetches with the selected page size and goes back to the first page", async () => {
      const user = userEvent.setup();
      render(<PerUserUsage {...defaultProps} />);
      await screen.findByText("user-1");
      await user.click(screen.getByTestId("pagination-next"));
      await screen.findByText("user-51");

      await user.click(screen.getByTestId("pagination-page-size"));
      await user.click(await screen.findByRole("option", { name: "100" }));

      expect(await screen.findByText("user-100")).toBeInTheDocument();
      expect(lastCall()).toEqual(["test-token", 1, 100, undefined]);
      expect(screen.getAllByRole("row")).toHaveLength(101);
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-100 of 120");
    });

    it("goes back to the first page when the tag filter changes", async () => {
      const user = userEvent.setup();
      const { rerender } = render(<PerUserUsage {...defaultProps} />);
      await screen.findByText("user-1");
      await user.click(screen.getByTestId("pagination-next"));
      await screen.findByText("user-51");
      const callsBeforeTagChange = mockPerUserAnalyticsCall.mock.calls.length;

      rerender(<PerUserUsage {...defaultProps} selectedTags={["curl/8.0"]} />);

      await waitFor(() => {
        expect(lastCall()).toEqual(["test-token", 1, 50, ["curl/8.0"]]);
      });
      expect(mockPerUserAnalyticsCall.mock.calls.slice(callsBeforeTagChange)).toEqual([
        ["test-token", 1, 50, ["curl/8.0"]],
      ]);
      expect(await screen.findByText("user-1")).toBeInTheDocument();
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-50 of 120");
    });

    it("does not request anything without an access token", () => {
      render(<PerUserUsage {...defaultProps} accessToken={null} />);

      expect(mockPerUserAnalyticsCall).not.toHaveBeenCalled();
      expect(screen.getByText("No per-user usage data")).toBeInTheDocument();
    });
  });

  it("renders the usage distribution as a stacked bar chart with the explicit palette and users formatter", async () => {
    render(<PerUserUsage {...defaultProps} />);

    await waitFor(() => {
      expect(mockPerUserAnalyticsCall).toHaveBeenCalled();
    });

    fireEvent.click(screen.getByText("Usage Distribution"));

    await waitFor(() => {
      expect(distributionChart().querySelectorAll("path.recharts-rectangle")).toHaveLength(4);
    });

    const chart = distributionChart();
    expect(chart.querySelectorAll(".recharts-bar")).toHaveLength(2);

    const rectangles = Array.from(chart.querySelectorAll("path.recharts-rectangle"));
    const fills = new Set(rectangles.map((rect) => rect.getAttribute("fill")));
    expect(fills).toEqual(new Set(["var(--color-blue-500, #3b82f6)", "var(--color-green-500, #22c55e)"]));

    const xPositions = new Set(rectangles.map((rect) => rect.getAttribute("d")?.match(/^M\s*([\d.]+)/)?.[1]));
    expect(xPositions.size).toBe(3);

    expect(chart).toHaveTextContent("curl/8.0");
    expect(chart).toHaveTextContent("Unknown");
    for (const bucket of [
      "1-9 requests",
      "10-99 requests",
      "100-999 requests",
      "1K-9.9K requests",
      "10K-99.9K requests",
      "100K+ requests",
    ]) {
      expect(chart).toHaveTextContent(bucket);
    }

    const tickTexts = Array.from(chart.querySelectorAll(".recharts-cartesian-axis-tick-value")).map(
      (tick) => tick.textContent ?? "",
    );
    expect(tickTexts.some((tick) => / users$/.test(tick))).toBe(true);
  });
});
