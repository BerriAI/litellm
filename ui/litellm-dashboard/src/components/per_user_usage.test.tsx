import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactElement, ReactNode } from "react";
import { afterEach, describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import PerUserUsage from "./per_user_usage";
import * as networking from "./networking";

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

const renderAtUrl = (ui: ReactElement, searchParams: string, onUrlUpdate: OnUrlUpdateFunction) =>
  render(ui, {
    wrapper: ({ children }: { children: ReactNode }) => (
      <NuqsTestingAdapter
        searchParams={searchParams}
        onUrlUpdate={onUrlUpdate}
        hasMemory
        resetUrlUpdateQueueOnMount={false}
      >
        {children}
      </NuqsTestingAdapter>
    ),
  });

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
    renderWithProviders(<PerUserUsage {...defaultProps} />);

    await waitFor(() => {
      expect(mockPerUserAnalyticsCall).toHaveBeenCalled();
    });

    expect(screen.getByText("Per User Usage")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText("u1")).toBeInTheDocument();
    });
  });

  it("keeps both tab panels mounted so switching tabs does not reset their state", async () => {
    renderWithProviders(<PerUserUsage {...defaultProps} />);

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

    const serveUsers = (total: number, delayMs = 0) => {
      mockPerUserAnalyticsCall.mockImplementation(async (_token, page = 1, pageSize = 50) => {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
        return {
          results: pageOfUsers(page, pageSize, total),
          total_count: total,
          page,
          page_size: pageSize,
          total_pages: Math.ceil(total / pageSize),
        };
      });
    };

    beforeEach(() => {
      serveUsers(TOTAL_USERS);
    });

    afterEach(() => {
      vi.restoreAllMocks();
    });

    const lastCall = () => mockPerUserAnalyticsCall.mock.calls[mockPerUserAnalyticsCall.mock.calls.length - 1];

    it("renders every row the server returns and shows the range from total_count", async () => {
      renderWithProviders(<PerUserUsage {...defaultProps} />);

      expect(await screen.findByText("user-50")).toBeInTheDocument();
      expect(screen.getByText("user-1")).toBeInTheDocument();
      expect(screen.getAllByRole("row")).toHaveLength(51);
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-50 of 120");
      expect(screen.getByTestId("pagination-prev")).toBeDisabled();
      expect(screen.getByTestId("pagination-next")).toBeEnabled();
    });

    it("refetches the next page when Next is clicked", async () => {
      const user = userEvent.setup();
      renderWithProviders(<PerUserUsage {...defaultProps} />);
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
      renderWithProviders(<PerUserUsage {...defaultProps} />);
      await screen.findByText("user-1");

      await user.click(screen.getByTestId("pagination-last"));

      expect(await screen.findByText("user-120")).toBeInTheDocument();
      expect(lastCall()).toEqual(["test-token", 3, 50, undefined]);
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 101-120 of 120");
      expect(screen.getByTestId("pagination-next")).toBeDisabled();
    });

    it("falls back to the last existing page when the data shrinks under the current page", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<PerUserUsage {...defaultProps} />, { onUrlUpdate });
      await screen.findByText("user-1");
      await user.click(screen.getByTestId("pagination-next"));
      await screen.findByText("user-51");
      const urlUpdatesBeforeShrink = onUrlUpdate.mock.calls.length;

      serveUsers(60);
      await user.click(screen.getByTestId("pagination-next"));

      await waitFor(() => {
        expect(onUrlUpdate.mock.calls.length).toBeGreaterThan(urlUpdatesBeforeShrink);
        expect(lastUrl(onUrlUpdate)?.get("per_user_page")).toBe("2");
      });
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
      renderWithProviders(<PerUserUsage {...defaultProps} />);
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
      renderWithProviders(<PerUserUsage {...defaultProps} />);
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
      const { rerender } = renderWithProviders(<PerUserUsage {...defaultProps} />);
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
      await screen.findByText("user-1");
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-50 of 120");
    });

    it("keeps the current page when the parent re-renders with the same tags", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      const { rerender } = renderWithProviders(<PerUserUsage {...defaultProps} selectedTags={["curl/8.0"]} />, {
        onUrlUpdate,
      });
      await screen.findByText("user-1");
      await user.click(screen.getByTestId("pagination-next"));
      await screen.findByText("user-51");
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("per_user_page")).toBe("2"));
      const urlUpdatesBeforeRerender = onUrlUpdate.mock.calls.length;

      rerender(<PerUserUsage {...defaultProps} selectedTags={["curl/8.0"]} />);

      await waitFor(() => expect(lastCall()).toEqual(["test-token", 2, 50, ["curl/8.0"]]));
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 51-100 of 120");
      expect(onUrlUpdate).toHaveBeenCalledTimes(urlUpdatesBeforeRerender);
    });

    it("does not request anything without an access token", () => {
      renderWithProviders(<PerUserUsage {...defaultProps} accessToken={null} />);

      expect(mockPerUserAnalyticsCall).not.toHaveBeenCalled();
      expect(screen.getByText("No per-user usage data")).toBeInTheDocument();
    });

    it("starts on the page and page size named by ?per_user_page= and ?per_user_page_size=", async () => {
      serveUsers(TOTAL_USERS, 50);
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderAtUrl(<PerUserUsage {...defaultProps} />, "?per_user_page=2&per_user_page_size=25", onUrlUpdate);

      expect(await screen.findByText("user-26")).toBeInTheDocument();
      expect(mockPerUserAnalyticsCall.mock.calls).toEqual([["test-token", 2, 25, undefined]]);
      expect(screen.getAllByRole("row")).toHaveLength(26);
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 26-50 of 120");
      expect(onUrlUpdate).not.toHaveBeenCalled();
    });

    it("keeps a deep-linked ?per_user_page= when the first request fails", async () => {
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
      mockPerUserAnalyticsCall.mockRejectedValue(new Error("upstream unavailable"));
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderAtUrl(<PerUserUsage {...defaultProps} />, "?per_user_page=3", onUrlUpdate);

      await waitFor(() =>
        expect(consoleError).toHaveBeenCalledWith("Failed to fetch per-user data:", expect.any(Error)),
      );
      await new Promise((resolve) => setTimeout(resolve, 20));

      expect(mockPerUserAnalyticsCall.mock.calls).toEqual([["test-token", 3, 50, undefined]]);
      expect(onUrlUpdate).not.toHaveBeenCalled();
      expect(screen.getByText("Page 3 of 1")).toBeInTheDocument();
    });

    it("moves a deep-linked page past the end back to the last page once a retry succeeds", async () => {
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
      serveUsers(60);
      mockPerUserAnalyticsCall.mockRejectedValueOnce(new Error("upstream unavailable"));
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      const { rerender } = renderAtUrl(<PerUserUsage {...defaultProps} />, "?per_user_page=3", onUrlUpdate);
      await waitFor(() => expect(consoleError).toHaveBeenCalledTimes(1));

      rerender(<PerUserUsage {...defaultProps} accessToken="refreshed-token" />);

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("per_user_page")).toBe("2"));
      expect(await screen.findByText("user-60")).toBeInTheDocument();
      expect(mockPerUserAnalyticsCall).toHaveBeenLastCalledWith("refreshed-token", 2, 50, undefined);
    });

    it("writes ?per_user_page= when the page changes and ?per_user_page_size= when the size changes", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<PerUserUsage {...defaultProps} />, { onUrlUpdate });
      await screen.findByText("user-1");

      await user.click(screen.getByTestId("pagination-next"));
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("per_user_page")).toBe("2"));
      expect(await screen.findByText("user-51")).toBeInTheDocument();

      await user.click(screen.getByTestId("pagination-page-size"));
      await user.click(await screen.findByRole("option", { name: "100" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("per_user_page_size")).toBe("100"));
      expect(lastUrl(onUrlUpdate)?.has("per_user_page")).toBe(false);
    });

    it("drops ?per_user_page= when the tag filter changes so the new filter starts on the first page", async () => {
      serveUsers(TOTAL_USERS, 50);
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      const { rerender } = renderAtUrl(<PerUserUsage {...defaultProps} />, "?per_user_page=2", onUrlUpdate);
      await screen.findByText("user-51");
      expect(onUrlUpdate).not.toHaveBeenCalled();

      rerender(<PerUserUsage {...defaultProps} selectedTags={["curl/8.0"]} />);

      await waitFor(() => expect(lastCall()).toEqual(["test-token", 1, 50, ["curl/8.0"]]));
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.has("per_user_page")).toBe(false));
      expect(await screen.findByText("user-1")).toBeInTheDocument();
    });
  });

  describe("tab URL state", () => {
    it("opens the tab named by ?per_user_tab=", async () => {
      renderWithProviders(<PerUserUsage {...defaultProps} />, { searchParams: "?per_user_tab=distribution" });

      expect(await screen.findByRole("tab", { name: "Usage Distribution" })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByRole("tab", { name: "User Details" })).toHaveAttribute("aria-selected", "false");
    });

    it("falls back to User Details and clears ?per_user_tab= when it names an unknown tab", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderAtUrl(<PerUserUsage {...defaultProps} />, "?per_user_tab=heatmap&view=user-agent-activity", onUrlUpdate);

      expect(screen.getByRole("tab", { name: "User Details" })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByRole("tab", { name: "Usage Distribution" })).toHaveAttribute("aria-selected", "false");
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.has("per_user_tab")).toBe(false));
      expect(lastUrl(onUrlUpdate)?.get("view")).toBe("user-agent-activity");
    });

    it("writes ?per_user_tab= when a tab is clicked and drops it on the default tab", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<PerUserUsage {...defaultProps} />, { onUrlUpdate });

      await user.click(screen.getByRole("tab", { name: "Usage Distribution" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("per_user_tab")).toBe("distribution"));

      await user.click(screen.getByRole("tab", { name: "User Details" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.has("per_user_tab")).toBe(false));
    });
  });

  it("renders the usage distribution as a stacked bar chart with the explicit palette and users formatter", async () => {
    renderWithProviders(<PerUserUsage {...defaultProps} />);

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
