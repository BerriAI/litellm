import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, testQueryClient, waitFor } from "../../../tests/test-utils";
import ModelAlertsPage from "@/app/(dashboard)/alerts/page";

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: () => ({ data: [{ team_id: "support", team_alias: "Support" }] }),
}));
vi.mock("@/components/ModelDeprecationAlerts/useDeprecationAlerts", () => ({
  useDeprecationAlerts: () => ({
    scope: "test",
    range: { start_date: "2026-10-06", end_date: "2026-11-04" },
    isPending: false,
    isError: false,
    data: [
      {
        id: "retirement",
        usageModel: "openai/gpt-4-turbo",
        requests: 5000,
        lastUsedDate: "2026-11-04",
        lastDayRequests: 20,
        model: {
          model_name: "support-model",
          deprecation_date: "2026-10-23",
          days_until_deprecation: -12,
          status: "deprecated",
          successor_model: "replacement-model",
        },
      },
    ],
  }),
}));

const calls: { path: string; query: URLSearchParams; method: string; body: unknown }[] = [];
function apiResponse(data: unknown) {
  return new Response(JSON.stringify(data), { headers: { "Content-Type": "application/json" } });
}

describe("model migration impact", () => {
  beforeEach(() => {
    testQueryClient.clear();
    calls.length = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const url = new URL(request.url);
      const call: (typeof calls)[number] = {
        path: url.pathname,
        query: url.searchParams,
        method: request.method,
        body: request.method === "PUT" ? await request.clone().json() : null,
      };
      calls.push(call);
      if (url.pathname === "/alerting/model-deprecations/tags") return apiResponse([]);
      if (request.method === "PUT" && url.pathname === "/alerting/model-deprecations/state")
        return apiResponse({ status: "resolved" });
      if (url.pathname.endsWith("/aggregated/keys")) {
        const offset = Number(url.searchParams.get("offset") || 0);
        const rows = Array.from({ length: 50 }, (_, i) => ({
          api_key: `reporting-${offset + i}`,
          metadata: { key_alias: `support-key-${offset + i}`, team_id: "support", user_email: "owner@example.com" },
          metrics: { api_requests: 100, spend: 1 },
        }));
        const page = { api_keys: rows, total_api_keys: 120, offset, limit: 50 };
        return apiResponse(page);
      }
      if (url.pathname.endsWith("/team/daily/activity/aggregated"))
        return apiResponse({ results: [], metadata: { entity_total_api_keys: {} } });
      return apiResponse({});
    });
  });
  afterEach(() => vi.restoreAllMocks());
  it("should render the inbox with last seen and the recommendation supplied by the API", () => {
    renderWithProviders(<ModelAlertsPage />);
    expect(screen.getByRole("tab", { name: "Model deprecations", exact: true })).toBeInTheDocument();
    expect(screen.getByText("replacement-model")).toBeInTheDocument();
    expect(screen.getByText("2026-11-04")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Review impact →" })).toHaveAttribute(
      "href",
      "/ui/alerts?type=model-deprecations&model=openai%2Fgpt-4-turbo",
    );
  });
  it("should page through all affected keys instead of showing a top-ten sample", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ModelAlertsPage />, { searchParams: { model: "openai/gpt-4-turbo" } });
    expect(await screen.findByRole("link", { name: "support-key-0" })).toBeInTheDocument();
    expect(screen.getByText(/Showing 1-50 of 120/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Go to next page", exact: true }));
    await screen.findByRole("link", { name: "support-key-50" });
    expect(
      calls.some(
        (call) =>
          call.path.endsWith("/aggregated/keys") &&
          call.query.get("offset") === "50" &&
          call.query.get("model") === "openai/gpt-4-turbo",
      ),
    ).toBe(true);
  });
  it("should resolve the whole model using its global latest-day snapshot even in a team-filtered view", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ModelAlertsPage />, {
      searchParams: { model: "openai/gpt-4-turbo", team: "support" },
    });
    expect(await screen.findByRole("link", { name: "support-key-0" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Mark resolved", exact: true }));
    const expectedAcknowledgement = {
      model: "openai/gpt-4-turbo",
      deprecation_date: "2026-10-23",
      status: "resolved",
      last_seen_date: "2026-11-04",
      last_seen_requests: 20,
    };
    await waitFor(() =>
      expect(
        calls.find((call) => call.method === "PUT" && call.path === "/alerting/model-deprecations/state")?.body,
      ).toEqual(expectedAcknowledgement),
    );

    expect(
      calls.some(
        (call) => call.path === "/team/daily/activity/aggregated/keys" && call.query.get("team_ids") === "support",
      ),
    ).toBe(true);
  });
});
