import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, testQueryClient, within } from "../../../tests/test-utils";
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

  it("keeps deprecation email setup accessible from the inbox", async () => {
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
});
