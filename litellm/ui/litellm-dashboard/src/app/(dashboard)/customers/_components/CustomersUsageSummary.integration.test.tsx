import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { formatDate } from "@/components/networking";
import { relativeTimeOptions } from "@/components/shared/date_range_presets";
import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  EMPTY_DAILY_ACTIVITY_RESPONSE,
  type DailyActivityAggregatedResponse,
} from "@/components/UsagePage/dailyActivityApi";
import type { SpendMetrics } from "@/components/UsagePage/types";
import {
  act,
  fireEvent,
  renderWithProviders,
  screen,
  testQueryClient,
  waitFor,
  within,
} from "../../../../../tests/test-utils";
import CustomersPage from "../page";
import CustomersUsageSummary from "./CustomersUsageSummary";

const metrics = (spend: number, tokens: number): SpendMetrics => ({
  spend,
  api_requests: 1,
  successful_requests: 1,
  failed_requests: 0,
  total_tokens: tokens,
  prompt_tokens: tokens * 0.8,
  completion_tokens: tokens * 0.2,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const summary: DailyActivityAggregatedResponse = {
  results: [
    { date: "2026-11-01", metrics: metrics(1000, 1000000) },
    { date: "2026-11-02", metrics: metrics(500, 500000) },
  ],
  metadata: {
    ...EMPTY_DAILY_ACTIVITY_METADATA,
    total_spend: 1500,
    total_tokens: 1500000,
    total_prompt_tokens: 1200000,
    total_completion_tokens: 300000,
  },
};

const monthly: DailyActivityAggregatedResponse = {
  results: [
    {
      date: "2026-11-01",
      metrics: metrics(50, 50000),
      breakdown: { entities: { visible: { metrics: metrics(50, 50000), metadata: {} } } },
    },
  ],
  metadata: { ...EMPTY_DAILY_ACTIVITY_METADATA, total_spend: 50, total_tokens: 50000 },
};

const fetchMock = vi.fn<typeof fetch>();
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const requestUrl = (input: Parameters<typeof fetch>[0]) =>
  new URL(input instanceof Request ? input.url : String(input));
const summaryRequests = () =>
  fetchMock.mock.calls
    .map(([input]) => requestUrl(input))
    .filter((url) => url.searchParams.get("include_entity_breakdown") === "false");

describe("CustomersUsageSummary", () => {
  beforeEach(() => {
    testQueryClient.clear();
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockReset();
    fetchMock.mockImplementation(async (input) => {
      const url = requestUrl(input);
      if (url.pathname === "/customer/list") {
        return json(
          url.searchParams.has("search")
            ? []
            : [{ user_id: "visible", alias: "Visible customer", blocked: false, spend: 0 }],
        );
      }
      if (url.pathname === "/customer/daily/activity/aggregated") {
        return json(url.searchParams.get("include_entity_breakdown") === "false" ? summary : monthly);
      }
      return json({ error: "Unexpected request" }, 404);
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    testQueryClient.clear();
    vi.unstubAllGlobals();
  });

  it("should show all-customer last-30-day totals independently of the monthly table and search", async () => {
    renderWithProviders(<CustomersPage />);

    const overview = screen.getByRole("region", { name: "Customer usage" });
    expect(await within(overview).findByText("$1,500.00")).toBeInTheDocument();
    expect(within(overview).getByTitle("1,500,000 tokens")).toHaveTextContent("1.5M");
    expect(within(overview).getByText("Last 30 days")).toBeInTheDocument();
    expect(await screen.findByText("Visible customer")).toBeInTheDocument();
    expect(await screen.findByText("$50.00")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Spend this month" })).toBeInTheDocument();
    const request = summaryRequests()[0];
    const period = relativeTimeOptions.find((option) => option.shortLabel === "30d")!.getValue();
    expect(request.pathname).toBe("/customer/daily/activity/aggregated");
    expect(request.searchParams.get("start_date")).toBe(formatDate(period.from));
    expect(request.searchParams.get("end_date")).toBe(formatDate(period.to));
    expect(request.searchParams.has("end_user_ids")).toBe(false);
    expect(request.searchParams.has("page")).toBe(false);

    act(() => {
      fireEvent.change(screen.getByRole("textbox"), { target: { value: "missing" } });
    });

    await waitFor(() => expect(screen.queryByText("Visible customer")).not.toBeInTheDocument());
    expect(within(overview).getByText("$1,500.00")).toBeInTheDocument();
    expect(within(overview).getByTitle("1,500,000 tokens")).toHaveTextContent("1.5M");
    expect(summaryRequests()).toHaveLength(1);
  });

  it("should keep the totals pending instead of showing zero before the response arrives", () => {
    fetchMock.mockImplementation(() => new Promise<Response>(() => {}));
    renderWithProviders(<CustomersUsageSummary />);

    expect(screen.getByRole("region", { name: "Customer usage" })).toHaveAttribute("aria-busy", "true");
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
    expect(screen.queryByTitle("0 tokens")).not.toBeInTheDocument();
  });

  it("should show a failed summary request as an error and retry without presenting zero totals", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(json({ error: "Unavailable" }, 500));
    renderWithProviders(<CustomersUsageSummary />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load customer usage totals");
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByText("$1,500.00")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("should show zero totals only after an empty aggregate response succeeds", async () => {
    fetchMock.mockResolvedValue(json(EMPTY_DAILY_ACTIVITY_RESPONSE));
    renderWithProviders(<CustomersUsageSummary />);

    expect(await screen.findByText("$0.00")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Customer usage" })).toHaveAttribute("aria-busy", "false");
    expect(screen.getByTitle("0 tokens")).toHaveTextContent("0");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
