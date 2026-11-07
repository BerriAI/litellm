import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { fireEvent, renderWithProviders, screen, testQueryClient, waitFor } from "../../../tests/test-utils";
import UsageRules from "./UsageRules";
import { FAMILY_SETTINGS } from "@/components/Issues/issueFixtures";
import UsageAlertContext from "./UsageAlertContext";
import UsageRuleForm from "./UsageRuleForm";
import type { RuleInput, UsageRule } from "./usageAlertApi";

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
const defaultSummary = {
  total: 40,
  enabled: 40,
  evaluation_status_counts: { pending: 0, delayed: 0, complete: 40, insufficient_coverage: 0, insufficient_history: 0 },
};
const savedRule: UsageRule = {
  id: "custom-rule",
  name: "Saved rule",
  scope_type: "deployment",
  scope_id: null,
  metric: "spend",
  direction: "increase",
  multiplier: 2,
  baseline_days: 14,
  min_history_days: 8,
  min_requests: 17,
  min_value: 10,
  enabled: false,
  notification_channels: ["email"],
  email_recipients: ["oncall@example.com"],
  created_at: "2026-11-01T00:00:00Z",
  updated_at: "2026-11-01T00:00:00Z",
  created_by: "test-admin",
  is_default: false,
};
const calls: { path: string; method: string; body: unknown; query: URLSearchParams }[] = [];
let customRules: UsageRule[] = [];
const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });

function ruleResponse(body: RuleInput) {
  if (body.email_recipients?.includes("invalid-recipient")) return json({ detail: "Invalid email recipient" }, 422);
  return json({ id: "new-rule", ...body });
}

function modelsResponse(query: URLSearchParams) {
  if (query.get("search") === "") return json({ detail: "Search must not be empty" }, 422);
  const page = {
    data: [{ scope_id: "provider/recorded-model", label: "Recorded model" }],
    limit: 20,
    next_cursor: null,
    start_date: "2026-10-07",
    end_date: "2026-11-05",
  };
  return json(page);
}

describe("usage alerts API integration", () => {
  beforeEach(() => {
    testQueryClient.clear();
    calls.length = 0;
    customRules = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const url = new URL(request.url);
      const body = request.method === "GET" ? null : await request.clone().json();
      const call = { path: url.pathname, method: request.method, body, query: url.searchParams };
      calls.push(call);
      if (url.pathname === "/management/v1/users/me") return json({ user_id: "test-admin", user_role: "proxy_admin" });
      if (url.pathname === "/usage/alerts/incident-test") return json({ ...observation, status: "resolved" });
      if (url.pathname === "/issues/families") return json(FAMILY_SETTINGS);
      if (url.pathname === "/usage/alert-rules" && request.method === "POST") {
        return ruleResponse(body);
      }
      if (url.pathname === "/usage/alert-rules") {
        const page = {
          data: customRules,
          total: customRules.length,
          offset: 0,
          limit: 20,
          default_summary: defaultSummary,
        };
        return json(page);
      }
      if (url.pathname === "/v2/team/list") {
        const page = {
          teams: [{ team_id: "team-example", team_alias: "Platform team" }],
          page: 1,
          page_size: 20,
          total: 1,
          total_pages: 1,
        };
        return json(page);
      }
      if (url.pathname === "/usage/alert-models") {
        return modelsResponse(url.searchParams);
      }
      return json({ detail: "Unexpected request" }, 404);
    });
  });
  afterEach(() => vi.restoreAllMocks());

  it.each([
    { enabled: true, status: "delayed", label: "Catching up · resumes automatically" },
    { enabled: true, status: "insufficient_history", label: "Collecting a baseline" },
    { enabled: false, status: "delayed", label: "Disabled" },
  ] as const)(
    "shows $label for a custom rule without promising disabled checks will resume",
    async ({ enabled, status, label }) => {
      customRules = [{ ...savedRule, enabled, evaluation_status: status }];
      const user = userEvent.setup();
      renderWithProviders(<UsageRules accessToken="test-access-token" />);
      await user.click(await screen.findByRole("button", { name: "Settings" }));
      await user.click(screen.getByText("Custom rules", { selector: "summary" }));
      expect(await screen.findByText(label)).toBeInTheDocument();
      if (status !== "delayed" || !enabled) {
        expect(screen.queryByText(/Catching up/)).not.toBeInTheDocument();
      }
    },
  );

  it("shows saved evidence on the matching entity without recomputing from a current rule", async () => {
    renderWithProviders(<UsageAlertContext alertId="incident-test" scope="key" entityId="reporting-test-id" />);
    expect(await screen.findByText(/Spend was \$120.00 on 2026-11-04 UTC/)).toHaveTextContent(
      "baseline was $30.00 across 5 recorded days",
    );
    expect(screen.getByText(/Spend was/)).toHaveTextContent("The alert condition was at least $90.00");
    expect(calls.some((call) => call.path.includes("alert-rules"))).toBe(false);
  });

  it("does not attach an incident explanation to the wrong entity", async () => {
    renderWithProviders(<UsageAlertContext alertId="incident-test" scope="team" entityId="team-example" />);
    expect(await screen.findByText("This alert belongs to a different entity.")).toBeInTheDocument();
    expect(screen.queryByText(/Spend was/)).not.toBeInTheDocument();
  });

  it("creates a configurable team rule with notification settings through the public API", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UsageRules accessToken="test-access-token" />);
    await user.click(await screen.findByRole("button", { name: "Settings" }));
    await user.click(screen.getByText("Custom rules", { selector: "summary" }));
    await user.click(await screen.findByRole("button", { name: "New rule", exact: true }));
    fireEvent.change(screen.getByLabelText("Resource"), { target: { value: "team" } });
    await user.click(screen.getByRole("combobox", { name: "Team" }));
    await user.click(await screen.findByRole("option", { name: "Platform team" }));
    fireEvent.change(screen.getByLabelText("Metric"), { target: { value: "requests" } });
    fireEvent.change(screen.getByLabelText("Change"), { target: { value: "decrease" } });
    fireEvent.change(screen.getByLabelText("Threshold (×)"), { target: { value: "2" } });
    await user.click(screen.getByText("Advanced", { exact: true }));
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Team request drop" } });
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
      min_value: 1000,
      notification_channels: ["email"],
      email_recipients: ["oncall@example.com"],
    };
    await waitFor(() => expect(calls.find((call) => call.method === "POST")?.body).toMatchObject(expected));
  });

  it("creates a useful default without opening Advanced or listing forty default rules", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UsageRules accessToken="test-access-token" />);
    await user.click(await screen.findByRole("button", { name: "Settings" }));
    expect(await screen.findByRole("button", { name: "Edit Unexpected spend" })).toBeInTheDocument();
    expect(calls.some((call) => call.path === "/usage/alert-rules")).toBe(false);
    await user.click(screen.getByText("Custom rules", { selector: "summary" }));
    await user.click(screen.getByRole("button", { name: "New rule", exact: true }));
    expect(screen.getByLabelText("Name")).not.toBeVisible();
    await user.click(screen.getByRole("button", { name: "Save rule" }));
    const expected = {
      name: "Spend increase for Entire gateway",
      scope_type: "deployment",
      scope_id: null,
      metric: "spend",
      multiplier: 3,
      baseline_days: 7,
      min_history_days: 3,
      min_requests: 0,
      min_value: 10,
      notification_channels: [],
    };
    await waitFor(() => expect(calls.find((call) => call.method === "POST")?.body).toMatchObject(expected));
  });

  it("selects the exact reporting model through bounded options rather than a deployment alias", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UsageRules accessToken="test-access-token" />);
    await user.click(await screen.findByRole("button", { name: "Settings" }));
    await user.click(screen.getByText("Custom rules", { selector: "summary" }));
    await user.click(await screen.findByRole("button", { name: "New rule", exact: true }));
    fireEvent.change(screen.getByLabelText("Resource"), { target: { value: "model" } });
    await user.click(screen.getByRole("combobox", { name: "Model" }));
    await user.click(await screen.findByRole("option", { name: "Recorded model" }));
    await user.click(screen.getByRole("button", { name: "Save rule" }));
    const expected = { scope_type: "model", scope_id: "provider/recorded-model" };
    await waitFor(() => expect(calls.find((call) => call.method === "POST")?.body).toMatchObject(expected));
    expect(calls.find((call) => call.path === "/usage/alert-models")?.query.get("limit")).toBe("20");
    expect(calls.some((call) => call.path === "/v2/model/info")).toBe(false);
  });

  it("recovers from a rejected email recipient when email is disabled without losing the draft", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UsageRules accessToken="test-access-token" />);
    await user.click(await screen.findByRole("button", { name: "Settings" }));
    await user.click(screen.getByText("Custom rules", { selector: "summary" }));
    await user.click(await screen.findByRole("button", { name: "New rule", exact: true }));
    await user.click(screen.getByRole("checkbox", { name: "Email", exact: true }));
    fireEvent.change(screen.getByLabelText("Email recipients"), { target: { value: "invalid-recipient" } });
    await user.click(screen.getByRole("button", { name: "Save rule" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Invalid email recipient");
    await user.click(screen.getByRole("checkbox", { name: "Email", exact: true }));
    await user.click(screen.getByRole("checkbox", { name: "Email", exact: true }));
    expect(screen.getByLabelText("Email recipients")).toHaveValue("invalid-recipient");
    await user.click(screen.getByRole("checkbox", { name: "Email", exact: true }));
    await user.click(screen.getByRole("button", { name: "Save rule" }));
    const expected = { notification_channels: [], email_recipients: [] };
    await waitFor(() => expect(calls.filter((call) => call.method === "POST").at(-1)?.body).toMatchObject(expected));
    expect(await screen.findByRole("button", { name: "New rule", exact: true })).toBeInTheDocument();
  });

  it("preserves existing advanced values when editing a different metric", async () => {
    const user = userEvent.setup();
    const save = vi.fn();
    renderWithProviders(<UsageRuleForm rule={savedRule} pending={false} onSave={save} onCancel={() => {}} />);
    fireEvent.change(screen.getByLabelText("Metric"), { target: { value: "latency" } });
    await user.click(screen.getByRole("button", { name: "Save rule" }));
    const expected = {
      name: savedRule.name,
      metric: "latency",
      baseline_days: 14,
      min_history_days: 8,
      min_requests: 17,
      min_value: 10,
      enabled: false,
      notification_channels: ["email"],
      email_recipients: ["oncall@example.com"],
    };
    expect(save).toHaveBeenCalledWith(expect.objectContaining(expected));
  });

  it("opens Advanced when a hidden edited field is invalid", () => {
    renderWithProviders(<UsageRuleForm rule={savedRule} pending={false} onSave={() => {}} onCancel={() => {}} />);
    expect(screen.getByLabelText("Name")).not.toBeVisible();
    fireEvent.invalid(screen.getByLabelText("Name"));
    expect(screen.getByLabelText("Name")).toBeVisible();
  });

  it("keeps a built-in decrease rule's identity while allowing its threshold to change", async () => {
    const user = userEvent.setup();
    const save = vi.fn();
    renderWithProviders(
      <UsageRuleForm
        rule={{ ...savedRule, is_default: true, direction: "decrease" }}
        pending={false}
        onSave={save}
        onCancel={() => {}}
      />,
    );
    expect(screen.getByLabelText("Resource")).toBeDisabled();
    expect(screen.getByLabelText("Metric")).toBeDisabled();
    expect(screen.getByLabelText("Change")).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Threshold (×)"), { target: { value: "4" } });
    await user.click(screen.getByRole("button", { name: "Save rule" }));
    const expected = {
      scope_type: "deployment",
      scope_id: null,
      metric: "spend",
      direction: "decrease",
      multiplier: 4,
    };
    expect(save).toHaveBeenCalledWith(expect.objectContaining(expected));
  });
});
