import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { act, renderWithProviders, screen, testQueryClient, waitFor, within } from "../../../tests/test-utils";
import Alerts from "./Alerts";
import type { SavedAlert } from "./alertPresentation";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
let role = "proxy_admin";
const alert: SavedAlert = {
  id: "9007199254740993",
  type: "spend",
  resource_type: "key",
  resource_id: "a".repeat(64),
  resource_name: "Research agent",
  status: "open",
  window_start: "2026-01-08T00:00:00Z",
  window_end: "2026-01-09T00:00:00Z",
  created_at: "2026-01-09T03:00:00Z",
  updated_at: "2026-01-09T03:00:00Z",
  evidence: { value: 20, baseline: 2, unit: "usd", sample_size: 100 },
  history: [
    { date: "2026-01-01", value: 2 },
    { date: "2026-01-08", value: 21 },
  ],
};
const calls: { path: string; query: URLSearchParams; method: string; body: unknown }[] = [];
let status = "open";
let rejectWrite = false;
let rejectHistory = false;
let monitoring = {
  model_deprecations_enabled: true,
  enabled: true,
  alerts_used: 0,
  alert_limit: 5 as number | null,
  disabled_by_config: false,
};
let rejectMonitoring = false;
function response(data: unknown, code = 200) {
  return new Response(JSON.stringify(data), { status: code, headers: { "Content-Type": "application/json" } });
}

describe("Alerts", () => {
  beforeEach(() => {
    testQueryClient.clear();
    calls.length = 0;
    status = "open";
    rejectWrite = false;
    rejectHistory = false;
    rejectMonitoring = false;
    monitoring = {
      model_deprecations_enabled: true,
      enabled: true,
      alerts_used: 0,
      alert_limit: 5,
      disabled_by_config: false,
    };
    role = "proxy_admin";
    document.cookie = `token=${btoa(JSON.stringify({ alg: "none" }))}.${btoa(JSON.stringify({ key: "example-session", exp: 4102444800 }))}.example; path=/`;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const url = new URL(request.url);
      const body = request.method === "PATCH" ? await request.clone().json() : null;
      const call = { path: url.pathname, query: url.searchParams, method: request.method, body };
      calls.push(call);
      if (url.pathname === "/litellm/.well-known/litellm-ui-config") return response({});
      if (url.pathname === "/management/v1/users/me") return response({ user_id: "example-admin", user_role: role });
      if (url.pathname === "/model/deprecations") return response({ deprecated: [], imminent: [] });
      if (url.pathname === "/alerts/monitoring") {
        if (rejectMonitoring) return response({ detail: "Unavailable" }, 503);
        if (request.method === "PATCH") {
          if (rejectWrite) return response({ detail: "Unavailable" }, 503);
          monitoring = { ...monitoring, ...body };
        }
        return response(monitoring);
      }
      if (request.method === "PATCH") {
        if (rejectWrite) return response({ detail: "Unavailable" }, 503);
        status = body.status;
        return response({ ...alert, status });
      }
      if (url.pathname === `/alerts/${alert.id}`)
        return response({ ...alert, status, history: rejectHistory ? null : alert.history });
      if (url.pathname === "/alerts") {
        if (url.searchParams.has("cursor"))
          return response({ data: [{ ...alert, id: "7", resource_name: "Second key" }], next_cursor: null });
        return response({
          data: url.searchParams.get("status") === status ? [{ ...alert, status }] : [],
          next_cursor: "8",
        });
      }
      throw new Error(`Unexpected endpoint ${url.pathname}`);
    });
  });
  afterEach(() => {
    vi.restoreAllMocks();
    document.cookie = "token=; Max-Age=0; path=/";
  });

  it("shows saved evidence, daily gaps, and a scoped Activity link", async () => {
    renderWithProviders(<Alerts />, { searchParams: { alert: alert.id } });
    expect(
      await screen.findByRole("heading", { name: "Spend on Research agent rose to 10× its baseline" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/\$20.00 on/)).toBeInTheDocument();
    const chart = screen.getByRole("table", { name: "Daily spend values" });
    expect(within(chart).getByRole("row", { name: "2026-01-08 $21.00" })).toBeInTheDocument();
    expect(within(chart).getByRole("row", { name: "2026-01-02 Not recorded" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "View key activity" })).toHaveAttribute(
      "href",
      `/ui/api-keys?key=${alert.resource_id}&detail_tab=activity&activity_from=2026-01-01&activity_to=2026-01-08`,
    );
  });

  it("keeps saved evidence visible when daily history cannot be loaded", async () => {
    rejectHistory = true;
    renderWithProviders(<Alerts />, { searchParams: { alert: alert.id } });
    expect(await screen.findByText(/Daily history is unavailable/)).toBeInTheDocument();
    expect(screen.getByText(/\$20.00 on/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Mark resolved" })).toBeEnabled();
  });

  it("saves status and reloads the detail, retaining exact large IDs", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Alerts />, { searchParams: { alert: alert.id } });
    await user.click(await screen.findByRole("button", { name: "Mark resolved" }));
    expect(await screen.findByRole("button", { name: "Reopen" })).toBeInTheDocument();
    expect(calls.find((call) => call.method === "PATCH")).toMatchObject({
      path: `/alerts/${alert.id}`,
      body: { status: "resolved" },
    });
    await user.click(screen.getByRole("button", { name: "Reopen" }));
    expect(await screen.findByRole("button", { name: "Dismiss" })).toBeInTheDocument();
  });

  it("retains the open alert when a status write fails", async () => {
    rejectWrite = true;
    const user = userEvent.setup();
    renderWithProviders(<Alerts />, { searchParams: { alert: alert.id } });
    await user.click(await screen.findByRole("button", { name: "Dismiss" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not save status");
    expect(screen.getByRole("button", { name: "Mark resolved" })).toBeEnabled();
  });

  it("uses the selected type and cursor when loading more results", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Alerts />, { searchParams: { type: "spend" } });
    await user.click(await screen.findByRole("button", { name: "Load more" }));
    expect(await screen.findByRole("button", { name: /Spend on Second key/ })).toBeInTheDocument();
    expect(
      calls.filter((call) => call.path === "/alerts").map((call) => [call.query.get("type"), call.query.get("cursor")]),
    ).toEqual([
      ["spend", null],
      ["spend", "8"],
    ]);
    await user.click(screen.getByRole("tab", { name: "resolved" }));
    expect(await screen.findByText("No resolved alerts.")).toBeInTheDocument();
  });

  it("keeps alert email setup accessible from the inbox", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Alerts />);
    await user.click(await screen.findByRole("button", { name: "Set up email reminders" }));
    expect(screen.getByRole("link", { name: "Open Email Settings" })).toHaveAttribute(
      "href",
      "/ui/logging-and-alerts?tab=email-alerts",
    );
  });

  it("lets viewers read the evidence without write controls", async () => {
    role = "proxy_admin_viewer";
    renderWithProviders(<Alerts />, { searchParams: { alert: alert.id } });
    expect(await screen.findByRole("heading", { name: /Spend on Research agent/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Dismiss" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Mark resolved" })).not.toBeInTheDocument();
  });
  it("keeps notification opt-in separate from the model deprecation toggle", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Alerts />);
    expect(
      await screen.findByText(
        "Model deprecation alerts are unlimited. Performance monitoring includes 5 free alerts, then requires Enterprise.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Manage alerts" }));
    const toggle = await screen.findByRole("switch", { name: "Model deprecations" });
    expect(toggle).toBeChecked();
    await user.click(toggle);
    await waitFor(() => expect(toggle).not.toBeChecked());
    expect(calls.filter((call) => call.method === "PATCH")).toEqual([
      expect.objectContaining({ path: "/alerts/monitoring", body: { model_deprecations_enabled: false } }),
    ]);
    expect(screen.getByRole("switch", { name: "Performance monitoring" })).toBeChecked();
    expect(screen.getByText(/Opt in to Performance Alerts in email or Slack settings/)).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(await screen.findByRole("button", { name: /Spend on Research agent/ })).toBeInTheDocument();
  });

  it("defaults on and persists opting out through the monitoring API", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Alerts />);
    await userEvent.click(await screen.findByRole("button", { name: "Manage alerts" }));
    const toggle = await screen.findByRole("switch", { name: "Performance monitoring" });
    expect(toggle).toBeChecked();
    expect(screen.getByText("0 of 5 free alerts generated.")).toBeInTheDocument();
    await user.click(toggle);
    expect(await screen.findByText("Off. Saved alerts remain available.")).toBeInTheDocument();
    expect(toggle).not.toBeChecked();
    expect(calls.find((call) => call.method === "PATCH")).toMatchObject({
      path: "/alerts/monitoring",
      body: { enabled: false },
    });
    await user.click(toggle);
    expect(await screen.findByText("0 of 5 free alerts generated.")).toBeInTheDocument();
  });

  it("explains exhaustion without blocking saved alerts or model deprecations", async () => {
    monitoring.alerts_used = 5;
    const user = userEvent.setup();
    renderWithProviders(<Alerts />);
    await userEvent.click(await screen.findByRole("button", { name: "Manage alerts" }));
    expect(
      await screen.findByText("You’ve used all 5 free alerts. New performance alerts are paused."),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Upgrade to Enterprise" })).toHaveAttribute(
      "href",
      "https://www.litellm.ai/enterprise",
    );
    await user.click(screen.getByRole("switch", { name: "Performance monitoring" }));
    expect(await screen.findByText("Off. Saved alerts remain available.")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Upgrade to Enterprise" })).not.toBeInTheDocument();
  });

  it("keeps the saved toggle value when an older refresh finishes later", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Alerts />);
    await userEvent.click(await screen.findByRole("button", { name: "Manage alerts" }));
    const toggle = await screen.findByRole("switch", { name: "Performance monitoring" });
    const originalFetch = vi.mocked(globalThis.fetch).getMockImplementation()!;
    const staleResponse = response(monitoring);
    const pendingRefresh = Promise.withResolvers<Response>();
    vi.mocked(globalThis.fetch).mockImplementation((input, init) => {
      const request = input instanceof Request ? input : new Request(input, init);
      if (new URL(request.url).pathname === "/alerts/monitoring" && request.method === "GET") {
        return pendingRefresh.promise;
      }
      return originalFetch(input, init);
    });
    await act(async () => {
      void testQueryClient.invalidateQueries({ queryKey: ["alert-monitoring"] });
    });
    await user.click(toggle);
    expect(await screen.findByText("Off. Saved alerts remain available.")).toBeInTheDocument();
    await act(async () => {
      pendingRefresh.resolve(staleResponse);
      await pendingRefresh.promise;
    });
    await waitFor(() => expect(testQueryClient.isFetching({ queryKey: ["alert-monitoring"] })).toBe(0));
    expect(toggle).not.toBeChecked();
    expect(screen.getByText("Off. Saved alerts remain available.")).toBeInTheDocument();
  });

  it("shows Enterprise monitoring without a free limit", async () => {
    monitoring.alert_limit = null;
    monitoring.alerts_used = 5;
    renderWithProviders(<Alerts />);
    await userEvent.click(await screen.findByRole("button", { name: "Manage alerts" }));
    expect(await screen.findByText("On with Enterprise.")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Upgrade to Enterprise" })).not.toBeInTheDocument();
  });

  it("respects config opt-out and view-only permissions", async () => {
    monitoring.disabled_by_config = true;
    monitoring.alerts_used = 5;
    monitoring.enabled = false;
    renderWithProviders(<Alerts />);
    await userEvent.click(await screen.findByRole("button", { name: "Manage alerts" }));
    expect(await screen.findByText("Disabled in proxy settings.")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Upgrade to Enterprise" })).not.toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Performance monitoring" })).toHaveAttribute("aria-disabled", "true");
  });

  it("lets viewers inspect alert settings without changing them", async () => {
    role = "proxy_admin_viewer";
    renderWithProviders(<Alerts />);
    await userEvent.click(await screen.findByRole("button", { name: "Manage alerts" }));
    expect(await screen.findByText("0 of 5 free alerts generated.")).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Performance monitoring" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("switch", { name: "Model deprecations" })).toHaveAttribute("aria-disabled", "true");
  });

  it("keeps the saved preference when a monitoring change fails", async () => {
    rejectWrite = true;
    const user = userEvent.setup();
    renderWithProviders(<Alerts />);
    await userEvent.click(await screen.findByRole("button", { name: "Manage alerts" }));
    await user.click(await screen.findByRole("switch", { name: "Performance monitoring" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not save monitoring settings");
    expect(screen.getByRole("switch", { name: "Performance monitoring" })).toBeChecked();
  });

  it("can retry monitoring settings without hiding the inbox", async () => {
    rejectMonitoring = true;
    const user = userEvent.setup();
    renderWithProviders(<Alerts />);
    await userEvent.click(await screen.findByRole("button", { name: "Manage alerts" }));
    expect(await within(screen.getByRole("dialog")).findByRole("alert")).toHaveTextContent(
      "Could not load monitoring settings",
    );
    rejectMonitoring = false;
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("switch", { name: "Performance monitoring" })).toBeChecked();
  });
});
