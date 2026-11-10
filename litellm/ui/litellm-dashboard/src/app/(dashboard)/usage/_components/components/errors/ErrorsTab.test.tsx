import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import { uiHref } from "@/utils/uiHref";
import ErrorsTab from "./ErrorsTab";
import type { RequestErrorActivity } from "./errorsData";

vi.mock("@/components/shared/charts", () => ({
  STACKED_USAGE_PALETTE: ["#111", "#222", "#333"],
  LineChart: ({
    data,
    categories,
    showDots,
  }: {
    data: readonly unknown[];
    categories: readonly string[];
    showDots?: boolean;
  }) => (
    <div data-testid="rate-chart" data-dots={showDots ? "on" : "off"}>
      {data.length} points · {categories.join(",")}
    </div>
  ),
  StackedUsageChart: ({ data, series }: { data: readonly unknown[]; series: readonly string[] }) => (
    <div data-testid="status-code-chart">
      {data.length} days · {series.join(",")}
    </div>
  ),
}));

const activity: RequestErrorActivity = {
  total_successful_requests: 950,
  total_failed_requests: 50,
  by_date: [
    {
      date: "2026-10-07",
      successful_requests: 500,
      failed_requests: 20,
      client_errors: 20,
      server_errors: 0,
      by_status_code: [{ status_code: 429, failed_requests: 20 }],
    },
    {
      date: "2026-10-08",
      successful_requests: 450,
      failed_requests: 30,
      client_errors: 25,
      server_errors: 5,
      by_status_code: [
        { status_code: 429, failed_requests: 20 },
        { status_code: 401, failed_requests: 5 },
        { status_code: 503, failed_requests: 5 },
      ],
    },
  ],
  by_status_code: [
    { status_code: 429, failed_requests: 40 },
    { status_code: 401, failed_requests: 5 },
    { status_code: 503, failed_requests: 5 },
  ],
  by_key: [
    {
      id: "88aa",
      label: "batch-runner",
      api_requests: 200,
      failed_requests: 40,
      top_status_code: 429,
      top_status_code_requests: 38,
    },
  ],
  by_team: [],
  by_user: [
    {
      id: "u-1",
      label: "dev@example.com",
      api_requests: 50,
      failed_requests: 5,
      top_status_code: 401,
      top_status_code_requests: 5,
    },
  ],
  by_model: [],
};

describe("ErrorsTab", () => {
  it("shows the error rate, the status callouts, both charts and the status list", () => {
    renderWithProviders(<ErrorsTab activity={activity} loading={false} failed={false} />);

    expect(screen.getByText("5.0%")).toBeInTheDocument();
    expect(screen.getByText("50 of 1,000 requests")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Rate limited (429)" })).toHaveTextContent("40");
    expect(screen.getByRole("group", { name: "Auth failures (401/403)" })).toHaveTextContent("5");
    expect(screen.getByTestId("rate-chart")).toHaveTextContent("2 points · rate");
    expect(screen.getByTestId("rate-chart")).toHaveAttribute("data-dots", "off");
    expect(screen.getByTestId("status-code-chart")).toHaveTextContent("2 days · 429,401,503");
    expect(
      within(screen.getByRole("list", { name: "Status codes in the chart" })).getAllByRole("listitem"),
    ).toHaveLength(3);

    const statusList = screen.getByRole("list", { name: "Failed requests by status code" });
    const items = within(statusList).getAllByRole("listitem");
    expect(items[0]).toHaveTextContent("429");
    expect(items[0]).toHaveTextContent("Rate Limited");
    expect(items[0]).toHaveTextContent("40 · 80.0%");
    expect(items).toHaveLength(3);
  });

  it("switches the error rate chart to one line per status code", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ErrorsTab activity={activity} loading={false} failed={false} />);

    await user.click(
      within(screen.getByRole("radiogroup", { name: "Show" })).getByRole("radio", { name: "By status code" }),
    );

    expect(screen.getByTestId("rate-chart")).toHaveTextContent("2 points · 429,401,503");
  });

  it("draws markers when the range holds a single day, in both views", async () => {
    const user = userEvent.setup();
    const oneDay: RequestErrorActivity = { ...activity, by_date: activity.by_date.slice(0, 1) };
    renderWithProviders(<ErrorsTab activity={oneDay} loading={false} failed={false} />);

    expect(screen.getByTestId("rate-chart")).toHaveTextContent("1 points · rate");
    expect(screen.getByTestId("rate-chart")).toHaveAttribute("data-dots", "on");

    await user.click(
      within(screen.getByRole("radiogroup", { name: "Show" })).getByRole("radio", { name: "By status code" }),
    );

    expect(screen.getByTestId("rate-chart")).toHaveTextContent("1 points · 429,401,503");
    expect(screen.getByTestId("rate-chart")).toHaveAttribute("data-dots", "on");
  });

  it("tells a user-scoped view that errors are deployment-wide instead of showing other users' failures", () => {
    renderWithProviders(<ErrorsTab activity={activity} loading={false} failed={false} userScoped />);

    expect(screen.getByRole("status")).toHaveTextContent("Errors are deployment-wide");
    expect(screen.queryByTestId("rate-chart")).not.toBeInTheDocument();
    expect(screen.queryByRole("radiogroup", { name: "Show" })).not.toBeInTheDocument();
  });

  it("says so instead of drawing an empty status view when nothing failed", async () => {
    const user = userEvent.setup();
    const healthy: RequestErrorActivity = {
      ...activity,
      total_failed_requests: 0,
      by_date: activity.by_date.map((day) => ({
        ...day,
        failed_requests: 0,
        client_errors: 0,
        server_errors: 0,
        by_status_code: [],
      })),
      by_status_code: [],
    };
    renderWithProviders(<ErrorsTab activity={healthy} loading={false} failed={false} />);

    expect(screen.getByTestId("rate-chart")).toHaveTextContent("2 points · rate");
    await user.click(
      within(screen.getByRole("radiogroup", { name: "Show" })).getByRole("radio", { name: "By status code" }),
    );

    expect(screen.queryByTestId("rate-chart")).not.toBeInTheDocument();
    expect(screen.getAllByText("No failed requests in this range").length).toBeGreaterThan(0);
  });

  it("ranks callers and switches the grouping", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ErrorsTab activity={activity} loading={false} failed={false} />);

    const keys = screen.getByRole("table", { name: "Virtual keys ranked by failed requests" });
    const row = within(keys).getAllByRole("row")[1];
    expect(row).toHaveTextContent("batch-runner");
    expect(row).toHaveTextContent("88aa");
    expect(row).toHaveTextContent("20.0%");
    expect(row).toHaveTextContent("429 · 38");

    await user.click(screen.getByRole("radio", { name: "Teams" }));
    expect(screen.getByText("No teams with failed requests in this range")).toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: "Users" }));
    const users = screen.getByRole("table", { name: "Users ranked by failed requests" });
    expect(within(users).getAllByRole("row")[1]).toHaveTextContent("dev@example.com");

    expect(screen.getByRole("button", { name: /Inspect failed requests in Logs/ })).toHaveAttribute(
      "href",
      uiHref("logs"),
    );
  });

  it("does not present a loading range as empty", () => {
    renderWithProviders(<ErrorsTab activity={null} loading failed={false} />);

    expect(screen.queryByText(/No requests in this range/)).not.toBeInTheDocument();
    expect(screen.queryByText(/No failed requests in this range/)).not.toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("reports a failed fetch instead of zeros", () => {
    renderWithProviders(<ErrorsTab activity={null} loading={false} failed />);

    expect(screen.getByRole("alert")).toHaveTextContent("Error activity could not be loaded");
    expect(screen.queryByText("Failed requests")).not.toBeInTheDocument();
  });

  it("says when the range had no traffic", () => {
    const quiet: RequestErrorActivity = {
      ...activity,
      total_successful_requests: 0,
      total_failed_requests: 0,
      by_date: [],
      by_status_code: [],
      by_key: [],
      by_user: [],
    };
    renderWithProviders(<ErrorsTab activity={quiet} loading={false} failed={false} />);

    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.getByText("No requests in this range")).toBeInTheDocument();
  });
});
