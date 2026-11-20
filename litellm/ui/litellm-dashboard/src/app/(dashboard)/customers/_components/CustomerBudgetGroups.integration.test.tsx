import userEvent, { type UserEvent } from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import {
  chooseSelectOption,
  jsonResponse as json,
  renderWithProviders,
  testQueryClient,
  screen,
  fireEvent,
  waitFor,
  within,
} from "../../../../../tests/test-utils";
import CustomerBudgetGroups from "./CustomerBudgetGroups";
import CustomerSettings from "../settings/page";
import type { Organization } from "./customerHierarchyForms";
import type { Budget } from "./customerBudgetForm";

const organization: Organization = {
  customer_org_id: "org-1",
  alias: "Example",
  default_budget_template_id: "standard",
  created_at: "",
  updated_at: "",
  created_by: "",
  updated_by: "",
  spend: 0,
  budget_groups_revision: "a".repeat(64),
  member_count: 0,
  default_budgets: [
    { budget_id: "standard", name: "Standard", budget_limits: [{ budget_duration: "1mo", max_budget: 10 }] },
    { budget_id: "old", name: "Old", budget_limits: [{ budget_duration: "1mo", max_budget: 20 }] },
  ],
};
vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: () => ({
    data: { data: [{ id: "model-a", owned_by: "openai" }] },
    isLoading: false,
    isError: false,
  }),
}));
const fetchMock = vi.fn<typeof fetch>();
const onSaved = vi.fn();
const limitsFor = (name: string) => within(screen.getByRole("row", { name }));
const emptyOrganization = { ...organization, default_budget_template_id: null, default_budgets: [] };
const organizationWithBudgets = (...budgets: Budget[]) => ({
  ...organization,
  default_budgets: organization.default_budgets.map((group, index) => ({ ...group, ...budgets[index] })),
});

async function openLimitEditor(user: UserEvent, group: string) {
  await user.click(limitsFor(group).getByRole("button", { name: "+ Add Budget Window" }));
  return screen.findByRole("dialog", { name: "Additional spend limit (USD)" });
}

describe("CustomerBudgetGroups", () => {
  beforeEach(() => {
    testQueryClient.clear();
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockReset();
    onSaved.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  it.each([false, true])("saves rate-only edits through the existing group endpoint, global=%s", async (global) => {
    const budget = { max_budget: 25, budget_duration: "1mo" };
    const source = {
      ...organizationWithBudgets(budget, budget),
      default_budget_template_id: global ? null : "standard",
    };
    fetchMock.mockResolvedValue(json(source));
    renderWithProviders(
      global ? (
        <CustomerBudgetGroups organization={source} onSaved={onSaved} global />
      ) : (
        <CustomerBudgetGroups organization={source} onSaved={onSaved} />
      ),
    );
    const tokens = limitsFor("Standard").getByRole("spinbutton", { name: "Tokens per minute" });
    const requests = limitsFor("Standard").getByRole("spinbutton", { name: "Requests per minute" });
    fireEvent.change(tokens, { target: { value: "50000" } });
    fireEvent.change(requests, { target: { value: "0" } });
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    const request = fetchMock.mock.calls[0][0] as Request;
    expect(new URL(request.url).pathname).toBe(
      global
        ? "/management/v1/customer_organization_budget_groups"
        : "/management/v1/customer_organizations/org-1/default_budgets",
    );
    expect(await request.clone().json()).toEqual({
      expected_revision: organization.budget_groups_revision,
      default_budget_template: global ? null : "Standard",
      default_budgets: [
        { budget_id: "standard", name: "Standard", tpm_limit: 50000, rpm_limit: 0 },
        { budget_id: "old", name: "Old" },
      ],
    });
  });

  const catalogs = [
    ["Overall app budget groups", "Overall App Budget Groups", "/management/v1/app_budget_groups", "overall-app"],
    [
      "Organization budget groups",
      "Overall Organization Budget Groups",
      "/management/v1/customer_organization_budget_groups",
      "organization",
    ],
    [
      "App × customer organization budget groups",
      "Per-App Organization Budget Groups",
      "/management/v1/app_customer_organization_budget_groups",
      "app",
    ],
    ["Per App Member Budget Groups", "App × member", "/management/v1/app_customer_member_budget_groups", "app-member"],
  ] as const;

  const catalogCases = catalogs.flatMap(([title, tabTitle, endpoint, scope]) =>
    [true, false].map((modelOnly) => ({ title, tabTitle, endpoint, scope, modelOnly })),
  );
  it.each(catalogCases)("saves $title, model-only=$modelOnly", async (entry) => {
    const { title, tabTitle, endpoint, scope, modelOnly } = entry;
    const directEditor = modelOnly || scope === "app-member";
    const user = userEvent.setup();
    const { period, duration } = {
      "overall-app": { period: "No reset", duration: null },
      organization: { period: "/ month", duration: "1mo" },
      app: { period: "Custom period", duration: "7d" },
      "app-member": { period: "/ year", duration: "12mo" },
    }[scope];
    const name = modelOnly ? "Models" : "Starter";
    const budget = {
      allowed_models: ["model-a"],
      ...(!modelOnly && { max_budget: 0, budget_duration: duration, soft_budget: 0, tpm_limit: 100, rpm_limit: 0 }),
    };
    const saved = {
      ...emptyOrganization,
      default_budgets: [{ budget_id: modelOnly ? "models" : "new", name, ...budget }],
    };
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      if (request.method === "PATCH") expect(new URL(request.url).pathname).toBe(endpoint);
      return json(request.method === "PATCH" ? saved : emptyOrganization);
    });
    const view = renderWithProviders(
      directEditor ? (
        <CustomerBudgetGroups organization={emptyOrganization} onSaved={onSaved} global scope={scope} />
      ) : (
        <CustomerSettings />
      ),
      { searchParams: `?tab=${scope}` },
    );
    if (!directEditor) expect(screen.getByRole("tab", { name: tabTitle })).toHaveAttribute("aria-selected", "true");
    const editor = within(await screen.findByRole("form", { name: "Budget groups" }));
    await user.click(await editor.findByRole("button", { name: "Add budget group" }));
    fireEvent.change(editor.getByLabelText("Group name 1"), { target: { value: name } });
    if (!modelOnly) {
      fireEvent.change(editor.getByRole("spinbutton", { name: "Spend limit (USD)" }), { target: { value: "0" } });
      await chooseSelectOption(user, editor.getByRole("combobox", { name: "Reset period" }), period);
      if (period === "Custom period")
        fireEvent.change(editor.getByLabelText("Custom period"), { target: { value: duration } });
      fireEvent.change(editor.getByLabelText("Soft budget (USD)"), { target: { value: "0" } });
      fireEvent.change(editor.getByLabelText("Tokens per minute"), { target: { value: "100" } });
      fireEvent.change(editor.getByLabelText("Requests per minute"), { target: { value: "0" } });
    }
    await chooseSelectOption(user, editor.getByRole("combobox", { name: "Search gateway models" }), "model-a");
    await user.keyboard("{Escape}");
    expect(editor.queryByRole("radio")).not.toBeInTheDocument();
    expect(editor.queryByRole("combobox", { name: "New members join" })).not.toBeInTheDocument();
    if (scope === "app-member")
      expect(editor.queryByRole("combobox", { name: "New organizations join" })).not.toBeInTheDocument();
    await user.click(editor.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(editor.getByRole("button", { name: "Save changes" })).toBeDisabled());
    const request = fetchMock.mock.calls
      .map(([input]) => input as Request)
      .find((request) => request.method === "PATCH")!;
    expect(new URL(request.url).pathname).toBe(endpoint);
    expect(await request.clone().json()).toEqual({
      expected_revision: organization.budget_groups_revision,
      default_budget_template: null,
      default_budgets: [{ name, ...budget }],
    });
    if (directEditor) {
      expect(onSaved).toHaveBeenCalledOnce();
      view.rerender(<CustomerBudgetGroups organization={saved} onSaved={onSaved} global scope={scope} />);
      expect(screen.getByLabelText("model-a")).toBeInTheDocument();
      return;
    }
    const otherTitle =
      title === "Organization budget groups"
        ? "App × customer organization budget groups"
        : "Organization budget groups";
    const otherTab =
      scope === "organization" ? "Per-App Organization Budget Groups" : "Overall Organization Budget Groups";
    await user.click(screen.getByRole("tab", { name: otherTab }));
    expect(
      within(screen.getByRole("region", { name: otherTitle })).queryByDisplayValue("Starter"),
    ).not.toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: tabTitle }));
    await user.click(editor.getByRole("button", { name: "Remove Starter" }));
    expect(editor.getByRole("button", { name: "Save changes" })).toBeEnabled();
  });

  it("should save additions, edited limits, the new default and deletions together", async () => {
    const user = userEvent.setup();
    const originalWindow = { budget_duration: "1mo" as const, max_budget: 10, reset_at: "2027-01-01T00:00:00Z" };
    const groupBudget = {
      tpm_limit: 1000,
      rpm_limit: 20,
      allowed_models: ["model-a"],
      budget_limits: [originalWindow],
    };
    const source = organizationWithBudgets(groupBudget);
    fetchMock.mockImplementation(async () =>
      json({
        ...source,
        default_budgets: [...source.default_budgets, { budget_id: "created-id", name: "New" }],
      }),
    );
    renderWithProviders(<CustomerBudgetGroups organization={source} onSaved={onSaved} />);
    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
    fireEvent.change(limitsFor("Standard").getByRole("spinbutton", { name: "Additional spend limit (USD)" }), {
      target: { value: "15" },
    });
    const primary = within(await openLimitEditor(user, "Standard"));
    expect(primary.getByRole("combobox", { name: "Reset period" })).toHaveTextContent("/ year");
    expect(primary.getByRole("button", { name: "Add limit" })).toBeDisabled();
    fireEvent.change(primary.getByRole("spinbutton"), { target: { value: "0" } });
    await user.click(primary.getByRole("button", { name: "Add limit" }));
    expect(limitsFor("Standard").getByRole("spinbutton", { name: "Spend limit (USD)" })).toHaveValue(0);
    await user.click(screen.getByRole("combobox", { name: "Group 1 at the cap" }));
    await user.click(await screen.findByRole("option", { name: "Pause, allow increase requests" }));
    await user.click(screen.getByRole("button", { name: "Add budget group" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Group name 3" }), { target: { value: "New" } });
    fireEvent.change(limitsFor("New").getByRole("spinbutton", { name: "Spend limit (USD)" }), {
      target: { value: "30" },
    });
    await chooseSelectOption(user, limitsFor("New").getByRole("combobox", { name: "Reset period" }), "/ year");
    const monthly = within(await openLimitEditor(user, "New"));
    expect(monthly.getByRole("combobox", { name: "Reset period" })).toHaveTextContent("/ month");
    fireEvent.change(monthly.getByRole("spinbutton"), { target: { value: "0" } });
    await user.click(monthly.getByRole("button", { name: "Add limit" }));
    const custom = within(await openLimitEditor(user, "New"));
    fireEvent.change(custom.getByRole("spinbutton"), { target: { value: "5" } });
    const invalidPeriods = ["6days", "2mo", "6mo"];
    for (const period of [...invalidPeriods, "1mo", "30d", "monthly"]) {
      fireEvent.change(custom.getByLabelText("Custom period"), { target: { value: period } });
      expect(custom.getByRole("button", { name: "Add limit" })).toBeDisabled();
      if (invalidPeriods.includes(period))
        expect(custom.getByRole("alert")).toHaveTextContent("Use 6d, 2w, 1mo, or 12mo");
    }
    fireEvent.change(custom.getByLabelText("Custom period"), { target: { value: "7d" } });
    await user.click(custom.getByRole("button", { name: "Add limit" }));
    await user.click(limitsFor("New").getAllByRole("button", { name: "Remove limit" })[0]);
    await user.click(screen.getByRole("combobox", { name: "New members join" }));
    await user.click(await screen.findByRole("option", { name: "New" }));
    await user.click(screen.getByRole("button", { name: "Remove Old" }));
    await chooseSelectOption(
      user,
      limitsFor("New").getAllByRole("combobox", { name: "Reset period" })[0],
      "Custom period",
    );
    const primaryPeriod = limitsFor("New").getAllByRole("textbox", { name: "Custom period" })[0];
    fireEvent.change(primaryPeriod, { target: { value: "6days" } });
    expect(primaryPeriod).toBeInvalid();
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(onSaved).not.toHaveBeenCalled();
    fireEvent.change(primaryPeriod, { target: { value: "12mo" } });
    expect(primaryPeriod).toBeValid();
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    const requests = fetchMock.mock.calls.map(([request]) => {
      if (!(request instanceof Request)) throw new Error("Expected shared HTTP client request");
      return request;
    });
    expect(requests.map((request) => [request.method, new URL(request.url).pathname])).toEqual([
      ["PATCH", "/management/v1/customer_organizations/org-1/default_budgets"],
    ]);
    expect(await requests[0].clone().json()).toEqual({
      expected_revision: organization.budget_groups_revision,
      default_budget_template: "New",
      default_budgets: [
        {
          budget_id: "standard",
          name: "Standard",
          cap_policy: "pause_allow_increase",
          max_budget: 0,
          budget_duration: "12mo",
          budget_limits: [{ ...originalWindow, max_budget: 15 }],
        },
        {
          name: "New",
          cap_policy: "pause_until_reset",
          max_budget: 30,
          budget_duration: "12mo",
          budget_limits: [{ max_budget: 5, budget_duration: "7d" }],
        },
      ],
    });
  });

  it("keeps independent drafts when switching URL tabs and falls back from an unknown tab", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    fetchMock.mockImplementation(async () => json({ ...organization, default_budget_template_id: null }));
    renderWithProviders(<CustomerSettings />, { searchParams: "?tab=unknown&source=portal", onUrlUpdate });
    expect(screen.getByRole("tab", { name: "Overall Organization Budget Groups" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    const overall = within(await screen.findByRole("region", { name: "Organization budget groups" }));
    fireEvent.change(overall.getByLabelText("Group name 1"), { target: { value: "Overall draft" } });
    await user.click(screen.getByRole("tab", { name: "Per-App Organization Budget Groups" }));
    await waitFor(() =>
      expect(onUrlUpdate).toHaveBeenLastCalledWith(
        expect.objectContaining({
          queryString: "?tab=app&source=portal",
          options: expect.objectContaining({ history: "push" }),
        }),
      ),
    );
    const perApp = within(screen.getByRole("region", { name: "App × customer organization budget groups" }));
    expect(perApp.getByLabelText("Group name 1")).toHaveValue("Standard");
    fireEvent.change(perApp.getByLabelText("Group name 1"), { target: { value: "App draft" } });
    await user.click(screen.getByRole("tab", { name: "Overall Organization Budget Groups" }));
    expect(overall.getByLabelText("Group name 1")).toHaveValue("Overall draft");
    await user.click(screen.getByRole("tab", { name: "Per-App Organization Budget Groups" }));
    expect(perApp.getByLabelText("Group name 1")).toHaveValue("App draft");
    expect(fetchMock.mock.calls.every(([input]) => (input as Request).method === "GET")).toBe(true);
  });

  it("saves a default for new organizations and lets No limit clear it", async () => {
    const user = userEvent.setup();
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      const body = request.method === "PATCH" ? await request.clone().json() : null;
      return json({ ...organization, default_budget_template_id: body?.default_budget_template ? "standard" : null });
    });
    renderWithProviders(<CustomerSettings />);
    const defaults = await screen.findByRole("combobox", { name: "New organizations join" });
    expect(defaults).toHaveTextContent("No limit");
    await user.click(defaults);
    await user.click(await screen.findByRole("option", { name: "Standard" }));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled());
    expect(defaults).toHaveTextContent("Standard");
    await user.click(defaults);
    await user.click(await screen.findByRole("option", { name: "No limit" }));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled());
    expect(defaults).toHaveTextContent("No limit");
    const requests = fetchMock.mock.calls
      .map(([input]) => input as Request)
      .filter((request) => request.method === "PATCH");
    expect(requests).toHaveLength(2);
    expect(await requests[0].clone().json()).toMatchObject({ default_budget_template: "Standard" });
    expect(await requests[1].clone().json()).toMatchObject({ default_budget_template: null });
  });

  it.each([
    [false, 409],
    [true, 409],
    [false, 204],
    [true, 204],
  ] as const)("keeps the draft after rejected save (global %s, status %s)", async (global, status) => {
    const user = userEvent.setup();
    const error = status === 204 ? "Unable to load customer" : "A referenced template cannot be deleted.";
    const source = { ...organization, default_budget_template_id: global ? null : "standard" };
    fetchMock.mockImplementation(async () =>
      status === 204 ? new Response(null, { status }) : json({ detail: error }, status),
    );
    renderWithProviders(<CustomerBudgetGroups global={global} organization={source} onSaved={onSaved} />);
    fireEvent.change(limitsFor("Standard").getByRole("spinbutton", { name: "Additional spend limit (USD)" }), {
      target: { value: "15" },
    });
    await user.click(screen.getByRole("button", { name: "Remove Old" }));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(error);
    expect(limitsFor("Standard").getByRole("spinbutton", { name: "Additional spend limit (USD)" })).toHaveValue(15);
    expect(screen.queryByDisplayValue("Old")).not.toBeInTheDocument();
    expect(onSaved).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledOnce();
    fetchMock.mockImplementation(async () => json(organization));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
  });

  it("should automatically make the first added group the default", async () => {
    const user = userEvent.setup();
    fetchMock.mockImplementation(async () => json(organization));
    renderWithProviders(
      <CustomerBudgetGroups
        organization={{ ...organization, default_budgets: [], default_budget_template_id: null }}
        onSaved={onSaved}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Add budget group" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Group name 1" }), { target: { value: "First" } });
    fireEvent.change(limitsFor("First").getByRole("spinbutton", { name: "Spend limit (USD)" }), {
      target: { value: "10" },
    });
    expect(screen.getByRole("combobox", { name: "New members join" })).toHaveTextContent("First");
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    const request = fetchMock.mock.calls[0][0];
    expect(request).toBeInstanceOf(Request);
    expect(await (request as Request).clone().json()).toMatchObject({ default_budget_template: "First" });
  });

  it("should start with an empty editor and no pending changes for an organization without groups", () => {
    renderWithProviders(
      <CustomerBudgetGroups
        organization={{ ...organization, default_budgets: [], default_budget_template_id: null }}
        onSaved={onSaved}
      />,
    );
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add budget group" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
  });
});
