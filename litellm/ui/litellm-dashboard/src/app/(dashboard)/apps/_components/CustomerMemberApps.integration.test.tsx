import userEvent, { type UserEvent } from "@testing-library/user-event";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import {
  jsonResponse as json,
  renderWithProviders as render,
  screen,
  fireEvent,
  act,
  waitFor,
  testQueryClient,
  chooseSelectOption,
} from "../../../../../tests/test-utils";
import CustomerMemberApps from "./CustomerMemberApps";
import { loadOrganizationApp, organizationAppKey } from "./appApi";
import type { App, AppMember } from "./appForms";
import { appBudgetPolicyCases, saveAppBudget } from "./appTestUtils";

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: () => ({ data: { data: [] }, isLoading: false, isError: false }),
}));
const app: App = {
  app_id: "app-1",
  resource_type: "team",
  resource_id: "team-assistant",
  alias: "Assistant",
  spend: 100,
  created_at: "",
  updated_at: "",
  created_by: "",
  updated_by: "",
};
const fetchMock = vi.fn<typeof fetch>();
let requests: Request[];
beforeEach(() => {
  requests = [];
  testQueryClient.clear();
  registerBaseUrlGetter(() => "http://localhost");
  fetchMock.mockReset();
  vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => {
    const request = new Request(input, init);
    requests.push(request.clone());
    return fetchMock(request);
  });
});
afterEach(() => vi.unstubAllGlobals());

async function openBudget() {
  const user = userEvent.setup();
  render(<CustomerMemberApps organizationId="org-1" memberId="member-1" />);
  await user.click(await screen.findByRole("button", { name: "Edit member budget for Assistant" }));
  return user;
}

const saveBudget = (user: UserEvent) => saveAppBudget(user, requests, "Save member app budget");

function serveMemberBudget(budget: Partial<AppMember>, apps: readonly App[] = [app]) {
  fetchMock.mockImplementation(async (input) =>
    json(
      new URL((input as Request).url).pathname.endsWith("/apps")
        ? apps
        : { app_id: app.app_id, customer_member_id: "member-1", spend: 0, ...budget },
    ),
  );
}

describe("customer member apps", () => {
  it("shows scoped member app usage and opens it from the app name", async () => {
    const onAppSelect = vi.fn();
    fetchMock.mockImplementation(async (input, init) => {
      const url = new URL(new Request(input, init).url);
      if (url.pathname.endsWith("/apps")) return json([app]);
      if (url.pathname.endsWith("/daily/activity")) {
        expect(url.searchParams.get("app_id")).toBe(app.app_id);
        expect(url.searchParams.get("customer_member_ids")).toBe("member-1");
        const metrics = {
          spend: 2.5,
          total_tokens: 1500,
          prompt_tokens: 1000,
          completion_tokens: 500,
          api_requests: 3,
        };
        return json({
          results: [
            { date: url.searchParams.get("start_date"), metrics, breakdown: { entities: { "member-1": { metrics } } } },
          ],
          metadata: { has_more: false, total_pages: 1 },
        });
      }
      const memberBudget = {
        app_id: app.app_id,
        customer_member_id: "member-1",
        spend: 99,
        budget_source: "customer_default",
        effective_budget: { max_budget: 10, budget_duration: "1mo", tpm_limit: 1000, rpm_limit: 20 },
        budget_usage: [{ source: "scalar", max_budget: 10, spend: 2, budget_duration: "1mo", reset_at: null }],
      };
      return json(memberBudget);
    });
    const user = userEvent.setup();
    render(
      <CustomerMemberApps
        organizationId="org-1"
        memberId="member-1"
        onAppSelect={onAppSelect}
        usageContext={{
          scope: { org: "org-1", team: null, member: "member-1" },
          startDate: "2026-11-01",
          endDate: "2026-11-18",
          organizationSpend: 10,
        }}
      />,
    );
    expect(await screen.findByRole("button", { name: "1,500 total tokens. View breakdown" })).toHaveTextContent("1.5K");
    expect(screen.getByText("$2.00 / $10.00")).toBeInTheDocument();
    expect(screen.getByText("20% used")).toBeInTheDocument();
    expect(screen.getByText("$8.00 left")).toBeInTheDocument();
    expect(screen.getByText("1,000 TPM")).toBeInTheDocument();
    expect(screen.getByText("20 RPM")).toBeInTheDocument();
    expect(screen.getByText("25%")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Assistant", exact: true }));
    expect(onAppSelect).toHaveBeenCalledExactlyOnceWith(app.app_id);
  });

  it("should show a compact member budget row and save an app-specific permanent budget", async () => {
    serveMemberBudget({
      spend: 2.5,
      budget_source: "customer_default",
      effective_budget: { rpm_limit: 1 },
    });
    const user = userEvent.setup();
    render(<CustomerMemberApps organizationId="org-1" memberId="member-1" />);
    expect(await screen.findByText("$2.50")).toBeInTheDocument();
    expect(screen.getByText("Customer member default")).toBeInTheDocument();
    expect(screen.getAllByRole("columnheader")).toHaveLength(3);
    expect(screen.queryByText("$100.00")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Edit member budget for Assistant" }));
    expect(screen.getByRole("dialog", { name: "Edit member budget for Assistant" })).toBeInTheDocument();
    await chooseSelectOption(
      user,
      screen.getByRole("combobox", { name: "Budget source" }),
      "Enter custom limits instead",
    );
    act(() => {
      fireEvent.change(screen.getByRole("spinbutton", { name: "Monthly budget" }), { target: { value: "5" } });
      fireEvent.change(screen.getByRole("spinbutton", { name: "Yearly budget" }), { target: { value: "50" } });
    });
    await user.click(screen.getByRole("button", { name: "Save member app budget" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
    const request = requests.find((row) => row.method === "PUT");
    expect(new URL(request!.url).pathname).toBe("/management/v1/apps/app-1/customer_members/member-1");
    expect(await request?.json()).toEqual({
      budget: {
        max_budget: 5,
        budget_duration: "1mo",
        budget_limits: [{ max_budget: 50, budget_duration: "12mo" }],
      },
    });
  });

  it("should distinguish tiny positive member spend from zero", async () => {
    serveMemberBudget({ spend: 0.00001234, budget_source: "none", effective_budget: null });
    render(<CustomerMemberApps organizationId="org-1" memberId="member-1" />);
    expect(await screen.findByText("<$0.01")).toBeInTheDocument();
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
  });

  it("should identify the primary budget period alongside its inherited source", async () => {
    const budget: Partial<AppMember> = {
      spend: 99,
      budget_source: "app_default",
      effective_budget: { max_budget: 10, budget_duration: "1mo" },
      budget_usage: [{ source: "scalar", max_budget: 10, spend: 2, budget_duration: "1mo", reset_at: null }],
    };
    serveMemberBudget(budget);
    render(<CustomerMemberApps organizationId="org-1" memberId="member-1" />);
    expect(await screen.findByText("App member default · Monthly")).toBeInTheDocument();
    expect(screen.getByText("$2.00 of $10.00 used")).toBeInTheDocument();
    expect(screen.queryByText("$99.00")).not.toBeInTheDocument();
  });

  it("should clear an existing override to restore inherited member defaults", async () => {
    const budget: Partial<AppMember> = {
      budget_override_id: "override-1",
      budget_override: { max_budget: 5 },
      budget_source: "member_override",
      effective_budget: { max_budget: 5 },
    };
    serveMemberBudget(budget);
    const user = await openBudget();
    await chooseSelectOption(
      user,
      screen.getByRole("combobox", { name: "Budget source" }),
      /^Inherited from customer organization or app/,
    );
    expect(await saveBudget(user)).toEqual({ budget: null, budget_override: null });
  });

  it.each(appBudgetPolicyCases)(
    "should initialize and save a permanent budget while an increase is present $source",
    async ({ policy, expectedPatch }) => {
      serveMemberBudget({ ...policy, effective_budget: { max_budget: 20 } }, [
        { ...app, default_customer_member_budget: { rpm_limit: 99, allowed_models: ["inherited-model"] } },
      ]);
      const user = await openBudget();
      expect(screen.getByRole("spinbutton", { name: "Monthly budget" })).toHaveValue(5);
      fireEvent.change(screen.getByRole("spinbutton", { name: "Monthly budget" }), { target: { value: "7" } });
      expect(await saveBudget(user)).toEqual(expectedPatch);
    },
  );

  it("edits a private budget despite a cached inherited-policy error", async () => {
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      const path = new URL(request.url).pathname;
      if (path.endsWith("/customer_organizations/org-1")) return json({ detail: "Policy unavailable" }, 503);
      return json(
        path.endsWith("/apps")
          ? [app]
          : { app_id: app.app_id, customer_member_id: "member-1", spend: 0, budget: { rpm_limit: 10 } },
      );
    });
    await expect(
      testQueryClient.fetchQuery({
        queryKey: organizationAppKey(app.app_id, "org-1"),
        queryFn: () => loadOrganizationApp(app.app_id, "org-1"),
        retry: false,
      }),
    ).rejects.toThrow();
    const user = await openBudget();
    fireEvent.change(await screen.findByRole("spinbutton", { name: "Requests per minute" }), {
      target: { value: "11" },
    });
    expect(await saveBudget(user)).toEqual({ budget: { rpm_limit: 11 } });
  });

  it("does not resend untouched spend when the policy refreshes during editing", async () => {
    let maximum = 5;
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      return json(
        new URL(request.url).pathname.endsWith("/apps")
          ? [app]
          : {
              app_id: app.app_id,
              customer_member_id: "member-1",
              spend: 0,
              budget: { max_budget: maximum, budget_duration: "1mo", rpm_limit: 10 },
            },
      );
    });
    const user = await openBudget();
    maximum = 25;
    await act(() => testQueryClient.refetchQueries({ queryKey: ["app-member", app.app_id, "member-1"], exact: true }));
    fireEvent.change(screen.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "11" } });
    expect(await saveBudget(user)).toEqual({ budget: { rpm_limit: 11 } });
  });

  it("should not enable overwriting a member budget after a read failure", async () => {
    fetchMock.mockImplementation(async (input) =>
      new URL((input as Request).url).pathname.endsWith("/apps")
        ? json([app])
        : json({ detail: "Member budgets unavailable" }, 503),
    );
    render(<CustomerMemberApps organizationId="org-1" memberId="member-1" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Member budgets unavailable");
    expect(screen.queryByRole("button", { name: "Edit member budget for Assistant" })).not.toBeInTheDocument();
  });

  it("should show an empty state when there are no apps", async () => {
    fetchMock.mockResolvedValue(json([]));
    render(<CustomerMemberApps organizationId="org-1" memberId="member-1" />);
    expect(await screen.findByText("No app budgets are configured yet.")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });
  it.each(["customer", "app"])("edits inherited %s policy without copying an active temporary grant", async (scope) => {
    const base = {
      max_budget: 500,
      budget_duration: "1mo",
      tpm_limit: 10000,
      rpm_limit: 50,
      budget_limits: [{ budget_duration: "12mo", max_budget: 5000 }],
    };
    fetchMock.mockImplementation(async (input, init) => {
      const request = new Request(input, init);
      const path = new URL(request.url).pathname;
      if (path.endsWith("/apps"))
        return json([{ ...app, default_customer_member_budget: scope === "app" ? base : null }]);
      if (path.endsWith("/customer_organizations/org-1"))
        return scope === "customer"
          ? json({ effective_customer_member_budget: base })
          : json({ detail: "Not found" }, 404);
      const member = {
        app_id: app.app_id,
        customer_member_id: "member-1",
        budget: null,
        budget_id: null,
        budget_override: { max_budget: 900, tpm_limit: 20000 },
        budget_override_expires_at: "2027-01-01T00:00:00Z",
        effective_budget: { ...base, max_budget: 900, tpm_limit: 20000 },
        budget_source: "member_override",
        spend: 0,
      };
      return json(member);
    });
    const user = await openBudget();
    await user.click(await screen.findByRole("combobox", { name: "Budget source" }));
    await user.click(await screen.findByRole("option", { name: "Enter custom limits instead" }));
    expect(screen.getByRole("spinbutton", { name: "Monthly budget" })).toHaveValue(500);
    expect(screen.getByRole("spinbutton", { name: "Yearly budget" })).toHaveValue(5000);
    expect(screen.getByRole("spinbutton", { name: "Tokens per minute" })).toHaveValue(10000);
    fireEvent.change(screen.getByRole("spinbutton", { name: "Monthly budget" }), { target: { value: "750" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "" } });
    expect(await saveBudget(user)).toEqual({
      budget: { max_budget: 750, budget_duration: "1mo", rpm_limit: null },
    });
  });
});
