import { Profiler } from "react";
import { act, fireEvent, renderWithProviders, screen, testQueryClient, waitFor, within } from "@/../tests/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { components } from "@/lib/http/schema";
import PromptCachingRequestsTable from "./PromptCachingRequestsTable";
import type { DateRange } from "./useDailyActivityRange";

type CacheRequest = components["schemas"]["PromptCachingRequest"];
type RequestsResponse = components["schemas"]["PromptCachingRequestsResponse"];
const firstCursor = { start_time: "2026-09-01T11:59:59.123456Z", request_id: "first-boundary?&" };
const secondCursor = { start_time: firstCursor.start_time, request_id: "second-boundary" };
const fetchMock = vi.fn<typeof fetch>();
const dates = { from: new Date(2026, 8, 1, 12), to: new Date(2026, 8, 2, 12) };
const request = (overrides: Partial<CacheRequest> = {}): CacheRequest => ({
  request_id: "request-default",
  start_time: "2026-09-01T12:00:00Z",
  model: "cache-test-model",
  gateway_injected: true,
  cache_read_tokens: 0,
  cache_creation_tokens: 1000,
  spend: 0.0375,
  net_savings: -0.0075,
  ...overrides,
});
const response = (requests: CacheRequest[], nextCursor: RequestsResponse["next_cursor"] = null) => {
  const body: RequestsResponse = { requests, has_more: nextCursor !== null, next_cursor: nextCursor, page_size: 50 };
  return Response.json(body);
};
const lastQuery = () => new URL(String(fetchMock.mock.calls.at(-1)?.[0]), "http://localhost").searchParams;

describe("PromptCachingRequestsTable", () => {
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    testQueryClient.clear();
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
    vi.useRealTimers();
  });

  it("separates recorded injection from cache hits, retains write premiums and unknown savings, and links each request", async () => {
    const clientHit = {
      request_id: "client-hit",
      gateway_injected: false,
      cache_read_tokens: 10000,
      cache_creation_tokens: 0,
      net_savings: 0.27,
    };
    fetchMock.mockResolvedValue(
      response([
        request({ request_id: "injected/write?&", net_savings: -0.0075 }),
        request(clientHit),
        request({ request_id: "unknown-price", net_savings: null }),
        request({ request_id: "no-benefit", net_savings: 0 }),
      ]),
    );
    renderWithProviders(<PromptCachingRequestsTable accessToken="token-a" dateValue={dates} />);

    const table = await screen.findByRole("table", { name: "Prompt caching requests" });
    const write = within(table).getByRole("row", { name: /injected\/write/ });
    expect(within(write).getByText("Recorded")).toBeInTheDocument();
    expect(within(write).getByText("1,000")).toBeInTheDocument();
    expect(within(write).getByText("$0.0375")).toBeInTheDocument();
    expect(within(write).getByText("-$0.0075")).toBeInTheDocument();
    expect(within(write).getByText(new Date("2026-09-01T12:00:00Z").toLocaleString())).toBeInTheDocument();
    expect(within(write).getByText("cache-test-model")).toHaveAttribute("title", "cache-test-model");
    expect(within(write).getByRole("link")).toHaveAttribute("href", "/ui/logs?log_id=injected%2Fwrite%3F%26");

    const hit = within(table).getByRole("row", { name: /client-hit/ });
    expect(within(hit).getByText("Not recorded")).toBeInTheDocument();
    expect(within(hit).getByText("10,000")).toBeInTheDocument();
    expect(within(hit).getByText("$0.2700")).toBeInTheDocument();
    expect(within(table).getByRole("row", { name: /unknown-price/ })).toHaveTextContent("Unavailable");
    expect(within(table).getByRole("row", { name: /no-benefit/ })).toHaveTextContent("$0.00");
    expect(screen.getByText(/after cache-write premiums/)).toBeInTheDocument();
    expect(lastQuery().get("start_date")).toBe("2026-09-01T00:00:00.000Z");
    expect(lastQuery().get("end_date")).toBe("2026-09-02T23:59:59.999Z");
    expect(fetchMock.mock.calls[0][1]?.headers).toEqual(expect.objectContaining({ Authorization: "Bearer token-a" }));
  });

  it("forwards complete server cursors, goes back to prior cursors, and clears them for each caching filter", async () => {
    fetchMock.mockImplementation(async (input) => {
      const query = new URL(String(input), "http://localhost").searchParams;
      const pages = new Map([
        [null, 1],
        [firstCursor.request_id, 2],
        [secondCursor.request_id, 3],
      ]);
      const page = pages.get(query.get("cursor_request_id"));
      const nextCursor =
        new Map([
          [1, firstCursor],
          [2, secondCursor],
        ]).get(page ?? 0) ?? null;
      return response([request({ request_id: `${query.get("filter")}-${page}` })], nextCursor);
    });
    renderWithProviders(<PromptCachingRequestsTable accessToken="token-a" dateValue={dates} />);
    await screen.findByRole("link", { name: "all-1" });
    expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled();
    expect(lastQuery().has("page")).toBe(false);
    expect(lastQuery().has("cursor_request_id")).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByRole("link", { name: "all-2" });
    expect(screen.getByText("Page 2")).toBeInTheDocument();
    expect(lastQuery().get("cursor_start_time")).toBe(firstCursor.start_time);
    expect(lastQuery().get("cursor_request_id")).toBe(firstCursor.request_id);
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByRole("link", { name: "all-3" });
    expect(screen.getByText("Page 3")).toBeInTheDocument();
    expect(lastQuery().get("cursor_start_time")).toBe(secondCursor.start_time);
    expect(lastQuery().get("cursor_request_id")).toBe(secondCursor.request_id);
    expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();

    await testQueryClient.invalidateQueries({ refetchType: "none" });
    fireEvent.click(screen.getByRole("button", { name: "Previous" }));
    await screen.findByRole("link", { name: "all-2" });
    await waitFor(() => expect(lastQuery().get("cursor_request_id")).toBe(firstCursor.request_id));
    expect(lastQuery().get("cursor_start_time")).toBe(firstCursor.start_time);
    expect(screen.getByText("Page 2")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Previous" }));
    await screen.findByRole("link", { name: "all-1" });
    await waitFor(() => expect(lastQuery().has("cursor_request_id")).toBe(false));
    expect(lastQuery().has("cursor_start_time")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByRole("link", { name: "all-2" });

    fireEvent.click(screen.getByRole("tab", { name: "LiteLLM injected" }));
    await screen.findByRole("link", { name: "injected-1" });
    expect(screen.queryByRole("link", { name: "all-2" })).not.toBeInTheDocument();
    expect(lastQuery().get("filter")).toBe("injected");
    expect(lastQuery().has("cursor_request_id")).toBe(false);
    expect(lastQuery().has("cursor_start_time")).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByRole("link", { name: "injected-2" });
    fireEvent.click(screen.getByRole("tab", { name: "Cache hits" }));
    await screen.findByRole("link", { name: "hits-1" });
    expect(lastQuery().get("filter")).toBe("hits");
    expect(lastQuery().get("page_size")).toBe("50");
    expect(screen.getByText("Page 1")).toBeInTheDocument();
  });

  it("includes the current UTC day for a range ending today, matching the activity totals", async () => {
    vi.stubEnv("TZ", "America/Los_Angeles");
    vi.setSystemTime(new Date("2026-09-20T03:00:00Z"));
    fetchMock.mockResolvedValue(response([]));
    const today = { from: new Date(2026, 8, 19), to: new Date() };
    renderWithProviders(<PromptCachingRequestsTable accessToken="token-a" dateValue={today} />);

    await screen.findByText("No matching prompt caching requests in this range");
    expect(lastQuery().get("start_date")).toBe("2026-09-19T00:00:00.000Z");
    expect(lastQuery().get("end_date")).toBe("2026-09-20T23:59:59.999Z");
  });

  it.each(["date", "authentication"])(
    "hides every old-scope frame and resets pagination when %s changes",
    async (change) => {
      fetchMock.mockResolvedValueOnce(response([request({ request_id: "old-first" })], firstCursor));
      fetchMock.mockResolvedValueOnce(response([request({ request_id: "old-second" })]));
      const committedOldRows: boolean[] = [];
      const snapshot = () => {
        committedOldRows.push(screen.queryByRole("link", { name: "old-second" }) !== null);
      };
      const tree = (accessToken: string, dateValue: DateRange) => (
        <Profiler id="request-scope" onRender={snapshot}>
          <PromptCachingRequestsTable accessToken={accessToken} dateValue={dateValue} />
        </Profiler>
      );
      const { rerender } = renderWithProviders(tree("token-a", dates));
      await screen.findByRole("link", { name: "old-first" });
      fireEvent.click(screen.getByRole("button", { name: "Next" }));
      await screen.findByRole("link", { name: "old-second" });

      const pending = Promise.withResolvers<Response>();
      fetchMock.mockReturnValueOnce(pending.promise);
      committedOldRows.length = 0;
      rerender(
        tree(
          change === "authentication" ? "token-b" : "token-a",
          change === "date" ? { ...dates, to: new Date(2026, 8, 3) } : dates,
        ),
      );

      expect(screen.getByRole("status")).toHaveTextContent("Loading requests");
      expect(committedOldRows.length).toBeGreaterThan(0);
      expect(committedOldRows.every((visible) => !visible)).toBe(true);
      expect(lastQuery().has("cursor_request_id")).toBe(false);
      expect(lastQuery().has("cursor_start_time")).toBe(false);
      if (change === "date") {
        expect(lastQuery().get("end_date")).toBe("2026-09-03T23:59:59.999Z");
      } else {
        expect(fetchMock.mock.calls.at(-1)?.[1]?.headers).toEqual(
          expect.objectContaining({ Authorization: "Bearer token-b" }),
        );
      }

      pending.resolve(response([request({ request_id: "new-first" })]));
      await screen.findByRole("link", { name: "new-first" });
      expect(screen.getByText("Page 1")).toBeInTheDocument();
      expect(committedOldRows.every((visible) => !visible)).toBe(true);
    },
  );

  it("ignores a delayed response from the previous caching filter", async () => {
    const stale = Promise.withResolvers<Response>();
    const current = Promise.withResolvers<Response>();
    fetchMock.mockReturnValueOnce(stale.promise).mockReturnValueOnce(current.promise);
    renderWithProviders(<PromptCachingRequestsTable accessToken="token-a" dateValue={dates} />);
    fireEvent.click(screen.getByRole("tab", { name: "Cache hits" }));
    expect(lastQuery().get("filter")).toBe("hits");

    current.resolve(response([request({ request_id: "current-hit" })]));
    await screen.findByRole("link", { name: "current-hit" });
    await act(async () => {
      stale.resolve(response([request({ request_id: "stale-all" })], firstCursor));
      await stale.promise;
    });

    expect(screen.getByRole("link", { name: "current-hit" })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "stale-all" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
  });

  it("offers retry after a failed read and shows the empty state after it succeeds", async () => {
    fetchMock.mockRejectedValueOnce(new Error("offline"));
    fetchMock.mockResolvedValueOnce(response([]));
    renderWithProviders(<PromptCachingRequestsTable accessToken="token-a" dateValue={dates} />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load prompt caching requests");
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("No matching prompt caching requests in this range")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does not request data for an incomplete date range", async () => {
    renderWithProviders(<PromptCachingRequestsTable accessToken="token-a" dateValue={{ from: dates.from }} />);
    expect(screen.getByText("Select a date range to view requests")).toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    await waitFor(() => expect(fetchMock).not.toHaveBeenCalled());
  });
});
