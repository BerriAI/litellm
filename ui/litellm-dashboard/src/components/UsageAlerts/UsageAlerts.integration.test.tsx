import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { act, fireEvent, renderWithProviders, screen, testQueryClient, waitFor } from "../../../tests/test-utils";
import UsageAlertsPanel from "./UsageAlertsPanel";
import UsageAlertContext from "./UsageAlertContext";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    accessToken: "test-access-token",
    userId: "test-admin",
    userRole: "Admin",
    isAuthorized: true,
    isViewOnly: false,
  }),
}));

const observation = {
  id: "incident-test",
  rule_id: "rule-test",
  rule_name: "Daily spend",
  scope_type: "key",
  scope_id: "reporting-test-id",
  metric: "spend",
  direction: "increase",
  status: "open",
  day: "2026-11-04",
  observed_value: 120,
  baseline_value: 30,
  baseline_start: "2026-10-28",
  baseline_end: "2026-11-03",
  baseline_sample_days: 5,
  threshold: 90,
  first_seen: "2026-11-05T01:00:00Z",
  last_seen: "2026-11-05T01:00:00Z",
  activity_path:
    "/api-keys?key=reporting-test-id&detail_tab=activity&activity_from=2026-10-28&activity_to=2026-11-04&alert=incident-test",
};
const calls: { path: string; method: string; body: unknown; query: URLSearchParams }[] = [];
const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });

describe("usage alerts API integration", () => {
  beforeEach(() => {
    testQueryClient.clear();
    calls.length = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const url = new URL(request.url);
      const body = request.method === "GET" ? null : await request.clone().json();
      const call = { path: url.pathname, method: request.method, body, query: url.searchParams };
      calls.push(call);
      if (url.pathname === "/management/v1/users/me") return json({ user_id: "test-admin", user_role: "proxy_admin" });
      if (url.pathname === "/usage/alerts/incident-test") return json({ ...observation, status: "resolved" });
      if (url.pathname === "/usage/alerts") {
        const offset = Number(url.searchParams.get("offset") ?? 0);
        const data = offset === 0 ? [observation] : [{ ...observation, id: "second-page", scope_id: "second-key" }];
        const page = { data, total: 51, limit: 50, offset };
        return json(page);
      }
      if (url.pathname === "/usage/alert-rules" && request.method === "POST") return json({ id: "new-rule", ...body });
      if (url.pathname === "/usage/alert-rules") {
        const page = { data: [], total: 0, offset: 0, limit: 20 };
        return json(page);
      }
      return json({ detail: "Unexpected request" }, 404);
    });
  });
  afterEach(() => vi.restoreAllMocks());

  it("shows the saved evidence, links to exact Activity context, and pages server-side", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UsageAlertsPanel />);
    const link = await screen.findByRole("link", { name: "reporting-test-id" });
    expect(link).toHaveAttribute("href", `/ui${observation.activity_path}`);
    expect(screen.getByText("$120.00")).toBeInTheDocument();
    expect(screen.getByText("$30.00")).toBeInTheDocument();
    expect(screen.getByText("At least $90.00")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Go to next page", exact: true }));
    expect(await screen.findByRole("link", { name: "second-key" })).toBeInTheDocument();
    expect(calls.some((call) => call.path === "/usage/alerts" && call.query.get("offset") === "50")).toBe(true);
  });

  it("updates an incident through the public API", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UsageAlertsPanel />);
    await user.click(await screen.findByRole("button", { name: "Resolve", exact: true }));
    await waitFor(() => expect(calls.find((call) => call.method === "PATCH")?.body).toEqual({ status: "resolved" }));
    expect(calls.find((call) => call.method === "PATCH")?.path).toBe("/usage/alerts/incident-test");
  });

  it("hides cached global incidents when the viewer loses global spend access", async () => {
    renderWithProviders(<UsageAlertsPanel />);
    expect(await screen.findByRole("link", { name: "reporting-test-id" })).toBeInTheDocument();
    await act(async () => {
      testQueryClient.setQueriesData(
        { queryKey: ["currentUser"] },
        { user_id: "test-admin", user_role: "internal_user_viewer" },
      );
    });
    await waitFor(() => expect(screen.queryByRole("link", { name: "reporting-test-id" })).not.toBeInTheDocument());
  });

  it("shows saved evidence on the matching entity without recomputing from a current rule", async () => {
    renderWithProviders(<UsageAlertContext alertId="incident-test" scope="key" entityId="reporting-test-id" />);
    expect(await screen.findByText(/Spend was \$120.00 on 2026-11-04 UTC/)).toHaveTextContent(
      "baseline was $30.00 across 5 recorded days",
    );
    expect(calls.some((call) => call.path.includes("alert-rules"))).toBe(false);
  });

  it("does not attach an incident explanation to the wrong entity", async () => {
    renderWithProviders(<UsageAlertContext alertId="incident-test" scope="team" entityId="team-example" />);
    expect(await screen.findByText("This alert belongs to a different entity.")).toBeInTheDocument();
    expect(screen.queryByText(/Spend was/)).not.toBeInTheDocument();
  });

  it("creates a configurable team rule with notification settings through the public API", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UsageAlertsPanel />);
    await user.click(screen.getByRole("button", { name: "Alert rules" }));
    await user.click(await screen.findByRole("button", { name: "New rule", exact: true }));
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Team request drop" } });
    fireEvent.change(screen.getByLabelText("Scope"), { target: { value: "team" } });
    fireEvent.change(screen.getByLabelText("Entity ID"), { target: { value: "team-example" } });
    fireEvent.change(screen.getByLabelText("Metric"), { target: { value: "requests" } });
    fireEvent.change(screen.getByLabelText("Change"), { target: { value: "decrease" } });
    fireEvent.change(screen.getByLabelText("Multiplier"), { target: { value: "2" } });
    fireEvent.change(screen.getByLabelText("Baseline days"), { target: { value: "1" } });
    expect(screen.getByLabelText("Minimum history days")).toHaveValue(1);
    await user.click(screen.getByRole("checkbox", { name: "Email", exact: true }));
    fireEvent.change(screen.getByLabelText("Email recipients"), { target: { value: "oncall@example.com" } });
    await user.click(screen.getByRole("button", { name: "Save rule" }));
    const expected = {
      name: "Team request drop",
      scope_type: "team",
      scope_id: "team-example",
      metric: "requests",
      direction: "decrease",
      multiplier: 2,
      baseline_days: 1,
      min_history_days: 1,
      notification_channels: ["email"],
      email_recipients: ["oncall@example.com"],
    };
    await waitFor(() => expect(calls.find((call) => call.method === "POST")?.body).toMatchObject(expected));
  });
});
