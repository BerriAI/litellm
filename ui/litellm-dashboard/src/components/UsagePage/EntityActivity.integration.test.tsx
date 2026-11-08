import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { fireEvent, renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import { EMPTY_DAILY_ACTIVITY_METADATA, EMPTY_DAILY_ACTIVITY_RESPONSE } from "./dailyActivityApi";
import EntityActivity from "./EntityActivity";

const auth = vi.hoisted(() => ({ userRole: "Admin", userId: "viewer", accessToken: "test-session" }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => auth }));
const fetchMock = vi.fn<typeof fetch>();
const dates = "activity_from=2026-10-01&activity_to=2026-10-07&signal=incident-1";
const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
const urls = () => fetchMock.mock.calls.map(([url]) => new URL(String(url), "http://example.com"));

beforeEach(() => {
  auth.userRole = "Admin";
  fetchMock.mockReset().mockImplementation(async () => response(EMPTY_DAILY_ACTIVITY_RESPONSE));
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("requestIdleCallback", (callback: IdleRequestCallback) =>
    setTimeout(() => callback({ didTimeout: false, timeRemaining: () => 50 }), 0),
  );
});

afterEach(() => vi.unstubAllGlobals());

describe("entity activity", () => {
  it("updates date filters without losing the linked incident", async () => {
    const onUrlUpdate = vi.fn();
    const { container } = renderWithProviders(<EntityActivity scope="key" id="key-hash" />, {
      searchParams: dates,
      onUrlUpdate,
    });
    await screen.findByText("No activity in this date range.");
    fireEvent.click(container.querySelector('[data-slot="advanced-date-picker-trigger"]')!);
    fireEvent.change(screen.getByDisplayValue("2026-10-01"), { target: { value: "2026-10-03" } });
    await userEvent.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() => expect(urls().at(-1)?.searchParams.get("start_date")).toBe("2026-10-03"));
    expect(urls().at(-1)?.searchParams.get("api_key")).toBe("key-hash");
    expect(onUrlUpdate.mock.lastCall?.[0].searchParams.get("signal")).toBe("incident-1");
  });

  it.each([
    ["key", "key-hash", "user", "api_key"],
    ["team", "team-1", "team", "team_ids"],
    ["user", "user-1", "user", "user_id"],
    ["model", "provider/model-1", "user", "model"],
  ] as const)("loads only the %s and the linked UTC dates", async (scope, id, entity, filter) => {
    renderWithProviders(<EntityActivity scope={scope} id={id} />, { searchParams: dates });
    await screen.findByText("No activity in this date range.");
    expect(urls()).toHaveLength(1);
    const url = urls()[0];
    expect(url.pathname).toBe(`/${entity}/daily/activity/aggregated`);
    expect(url.searchParams.get(filter)).toBe(id);
    expect(url.searchParams.get("start_date")).toBe("2026-10-01");
    expect(url.searchParams.get("end_date")).toBe("2026-10-07");
    expect(url.searchParams.get("timezone")).toBe("0");
    expect(url.searchParams.get("bounded")).toBe("true");
    expect(url.searchParams.get("include_model_breakdown")).toBe(scope === "key" ? "true" : null);
    expect(url.searchParams.has("signal")).toBe(false);
  });

  it("keeps a member's key request within their user scope", async () => {
    auth.userRole = "Internal User";
    renderWithProviders(<EntityActivity scope="key" id="key-hash" />, { searchParams: dates });
    await screen.findByText("No activity in this date range.");
    expect(urls()[0].searchParams.get("user_id")).toBe("viewer");
  });

  it("loads a member's key through the authorized team scope without filtering to the viewer", async () => {
    auth.userRole = "Internal User";
    renderWithProviders(<EntityActivity scope="key" id="member-key" teamId="team-1" />, { searchParams: dates });
    await screen.findByText("No activity in this date range.");
    expect(urls()).toHaveLength(1);
    expect(urls()[0].pathname).toBe("/team/daily/activity/aggregated");
    expect(urls()[0].searchParams.get("team_ids")).toBe("team-1");
    expect(urls()[0].searchParams.get("api_key")).toBe("member-key");
    expect(urls()[0].searchParams.has("user_id")).toBe(false);
  });

  it("does not query global model activity for a member", async () => {
    auth.userRole = "Internal User";
    renderWithProviders(<EntityActivity scope="model" id="provider/model-1" />, { searchParams: dates });
    expect(screen.getByText("Model activity requires access to global usage.")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("does not query an inverted date range", () => {
    renderWithProviders(<EntityActivity scope="team" id="team-1" />, {
      searchParams: "activity_from=2026-10-07&activity_to=2026-10-01",
    });
    expect(screen.getByRole("alert")).toHaveTextContent("The start date must be on or before the end date.");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("shows a failed request and retries the same scope", async () => {
    fetchMock.mockImplementationOnce(async () => response({ error: "Forbidden" }, 403));
    renderWithProviders(<EntityActivity scope="team" id="team-1" />, { searchParams: dates });
    expect(await screen.findByRole("alert")).toHaveTextContent("Unable to load activity.");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    await screen.findByText("No activity in this date range.");
    expect(urls()).toHaveLength(2);
    expect(urls()[0].href).toBe(urls()[1].href);
  });

  it("uses the existing Usage statistics for the returned data", async () => {
    fetchMock.mockImplementation(async () =>
      response({
        results: [
          {
            date: "2026-10-01",
            metrics: {
              api_requests: 42,
              successful_requests: 40,
              failed_requests: 2,
              spend: 12,
              total_tokens: 4200,
              prompt_tokens: 4000,
              completion_tokens: 200,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0,
              total_response_time_ms: 8400,
              timed_requests: 42,
            },
            breakdown: {},
          },
        ],
        metadata: {
          ...EMPTY_DAILY_ACTIVITY_METADATA,
          total_api_requests: 42,
          total_successful_requests: 40,
          total_failed_requests: 2,
          total_spend: 12,
          total_tokens: 4200,
          total_response_time_ms: 8400,
          total_timed_requests: 42,
        },
      }),
    );
    renderWithProviders(<EntityActivity scope="team" id="team-1" />, { searchParams: dates });
    await screen.findByText("Total Requests");
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("4,200")).toBeInTheDocument();
    await waitFor(() => expect(urls()).toHaveLength(1));
  });
});
