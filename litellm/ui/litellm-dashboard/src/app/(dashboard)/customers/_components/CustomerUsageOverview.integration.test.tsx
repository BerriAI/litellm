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
import { renderWithProviders, screen, testQueryClient, waitFor } from "../../../../../tests/test-utils";
import CustomerUsageOverview from "./CustomerUsageOverview";

const metrics = (spend: number, requests: number, tokens: number): SpendMetrics => ({
  spend,
  api_requests: requests,
  successful_requests: requests,
  failed_requests: 0,
  total_tokens: tokens,
  prompt_tokens: tokens * 0.8,
  completion_tokens: tokens * 0.2,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const daily = (date: string, model: string, values: SpendMetrics) => ({
  date,
  metrics: values,
  breakdown: { model_groups: { [model]: { metrics: values, metadata: {} } } },
});

const activity: DailyActivityAggregatedResponse = {
  results: [
    daily("2026-11-01", "customer-primary", metrics(12, 5, 500)),
    daily("2026-11-02", "customer-reserve", metrics(3, 2, 200)),
  ],
  metadata: {
    ...EMPTY_DAILY_ACTIVITY_METADATA,
    total_spend: 15,
    total_api_requests: 7,
    total_successful_requests: 7,
    total_tokens: 700,
    total_prompt_tokens: 560,
    total_completion_tokens: 140,
  },
};

const fetchMock = vi.fn<typeof fetch>();
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const requestUrl = (input: Parameters<typeof fetch>[0]) =>
  new URL(input instanceof Request ? input.url : String(input));
const requestedUrls = () => fetchMock.mock.calls.map(([input]) => requestUrl(input));

describe("CustomerUsageOverview", () => {
  beforeEach(() => {
    testQueryClient.clear();
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockReset();
    fetchMock.mockResolvedValue(json(activity));
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("requestIdleCallback", (callback: IdleRequestCallback) => {
      callback({ didTimeout: false, timeRemaining: () => 50 });
      return 1;
    });
  });

  afterEach(() => {
    testQueryClient.clear();
    vi.unstubAllGlobals();
  });

  it("should render customer-scoped daily usage with the Usage page's totals and model breakdown", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomerUsageOverview customerId="customer-1" />);

    expect(await screen.findByText("$15.00")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Daily usage" })).toBeInTheDocument();
    expect(screen.getByTitle("700 tokens")).toBeInTheDocument();
    expect(screen.getByText("customer-primary")).toBeInTheDocument();
    expect(screen.getByText("customer-reserve")).toBeInTheDocument();
    expect(screen.queryByText("No budget set")).not.toBeInTheDocument();
    const url = requestedUrls()[0];
    expect(url.pathname).toBe("/customer/daily/activity/aggregated");
    expect(url.searchParams.get("end_user_ids")).toBe("customer-1");
    expect(url.searchParams.get("include_entity_breakdown")).toBe("false");
    const month = relativeTimeOptions.find((option) => option.shortLabel === "MTD")!.getValue();
    expect(url.searchParams.get("start_date")).toBe(formatDate(month.from));
    expect(url.searchParams.get("end_date")).toBe(formatDate(month.to));

    await user.click(screen.getByRole("radio", { name: "Tokens" }));
    expect(screen.getByText("500")).toBeInTheDocument();
    expect(screen.getByText("200")).toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: "Requests" }));
    expect(screen.getByText("5")).toBeInTheDocument();
  });

  it("should change the customer date range without losing the selected customer or unrelated URL state", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomerUsageOverview customerId="customer-1" />, {
      searchParams: "?id=customer-1&source=overview",
      onUrlUpdate,
    });
    expect(await screen.findByText("customer-primary")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: / - / }));
    await user.click(screen.getByRole("button", { name: /Last 7 days/ }));
    await user.click(screen.getByRole("button", { name: "Apply" }));

    const week = relativeTimeOptions.find((option) => option.shortLabel === "7d")!.getValue();
    await waitFor(() => expect(requestedUrls().at(-1)?.searchParams.get("start_date")).toBe(formatDate(week.from)));
    expect(requestedUrls().at(-1)?.searchParams.get("end_user_ids")).toBe("customer-1");
    expect(onUrlUpdate).toHaveBeenLastCalledWith(
      expect.objectContaining({ queryString: "?id=customer-1&source=overview&range=7d" }),
    );
  });

  it("should remove the previous customer's values while the next customer's usage loads", async () => {
    const { rerender } = renderWithProviders(<CustomerUsageOverview customerId="customer-1" />);
    expect(await screen.findByText("customer-primary")).toBeInTheDocument();
    fetchMock.mockImplementation(() => new Promise<Response>(() => {}));

    rerender(<CustomerUsageOverview customerId="customer-2" />);

    expect(screen.queryByText("customer-primary")).not.toBeInTheDocument();
    expect(screen.queryByText("$15.00")).not.toBeInTheDocument();
    expect(screen.getByRole("status", { name: "Loading top models" })).toBeInTheDocument();
    await waitFor(() => expect(requestedUrls().at(-1)?.searchParams.get("end_user_ids")).toBe("customer-2"));
  });

  it("should show a failed usage request as an error and let the customer retry", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(json({ error: "Unavailable" }, 500));
    renderWithProviders(<CustomerUsageOverview customerId="customer-1" />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load customer usage");
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
    expect(screen.queryByText("No usage in this range")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByText("$15.00")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("should show an empty range only after the customer query returns no activity", async () => {
    fetchMock.mockResolvedValue(json(EMPTY_DAILY_ACTIVITY_RESPONSE));
    renderWithProviders(<CustomerUsageOverview customerId="customer-1" />);

    expect(await screen.findByText("No usage in this range")).toBeInTheDocument();
    expect(screen.getByText("$0.00")).toBeInTheDocument();
    expect(screen.queryByRole("status", { name: "Loading top models" })).not.toBeInTheDocument();
  });
});
