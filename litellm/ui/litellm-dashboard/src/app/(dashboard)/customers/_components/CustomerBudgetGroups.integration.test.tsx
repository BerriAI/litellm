import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import {
  render,
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
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("CustomerBudgetGroups", () => {
  it.each(["organization", "app"] as const)("saves model-only %s groups and reloads their selection", async (scope) => {
    const user = userEvent.setup();
    const source = { ...organization, default_budget_template_id: null, default_budgets: [] };
    fetchMock.mockResolvedValue(
      json({
        ...source,
        default_budgets: [{ budget_id: "models", name: "Models", allowed_models: ["model-a"] }],
      }),
    );
    const view = render(<CustomerBudgetGroups organization={source} onSaved={onSaved} global scope={scope} />);
    await user.click(screen.getByRole("button", { name: "Add budget group" }));
    fireEvent.change(screen.getByLabelText("Group name 1"), { target: { value: "Models" } });
    await user.click(screen.getByRole("combobox", { name: "Search gateway models" }));
    await user.click(await screen.findByRole("option", { name: "model-a" }));
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    const request = fetchMock.mock.calls[0][0] as Request;
    expect(new URL(request.url).pathname).toBe(
      scope === "app"
        ? "/management/v1/app_customer_organization_budget_groups"
        : "/management/v1/customer_organization_budget_groups",
    );
    expect(await request.clone().json()).toMatchObject({
      default_budgets: [{ name: "Models", allowed_models: ["model-a"] }],
    });
    view.rerender(
      <CustomerBudgetGroups
        organization={{
          ...source,
          default_budgets: [{ budget_id: "models", name: "Models", allowed_models: ["model-a"] }],
        }}
        onSaved={onSaved}
        global
        scope={scope}
      />,
    );
    expect(screen.getByLabelText("model-a")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
  });

  beforeEach(() => {
    testQueryClient.clear();
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockReset();
    onSaved.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  it.each([false, true])("saves rate-only edits through the existing group endpoint, global=%s", async (global) => {
    const source = { ...organization, default_budget_template_id: global ? null : "standard" };
    fetchMock.mockResolvedValue(json(source));
    render(
      global ? (
        <CustomerBudgetGroups organization={source} onSaved={onSaved} global />
      ) : (
        <CustomerBudgetGroups organization={organization} onSaved={onSaved} />
      ),
    );
    const tokens = screen.getByRole("spinbutton", { name: "Group 1 tokens per minute" });
    const requests = screen.getByRole("spinbutton", { name: "Group 1 requests per minute" });
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

  it.each([
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
  ])("saves only the %s catalog in Settings", async (title, tabTitle, endpoint, tab) => {
    const user = userEvent.setup();
    const catalog = { ...organization, default_budget_template_id: null, default_budgets: [] };
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      if (request.method === "PATCH") expect(new URL(request.url).pathname).toBe(endpoint);
      return json(
        request.method === "PATCH"
          ? {
              ...catalog,
              default_budgets: [
                { budget_id: "new", name: "Starter", budget_limits: [{ budget_duration: "1mo", max_budget: 0 }] },
              ],
            }
          : catalog,
      );
    });
    renderWithProviders(<CustomerSettings />, { searchParams: `?tab=${tab}` });
    expect(screen.getByRole("tab", { name: tabTitle })).toHaveAttribute("aria-selected", "true");
    const editor = within(await screen.findByRole("tabpanel", { name: tabTitle }));
    await user.click(await editor.findByRole("button", { name: "Add budget group" }));
    fireEvent.change(editor.getByLabelText("Group name 1"), { target: { value: "Starter" } });
    fireEvent.change(editor.getByLabelText("Group 1 monthly budget (USD)"), { target: { value: "0" } });
    expect(editor.queryByRole("radio")).not.toBeInTheDocument();
    expect(editor.queryByRole("combobox", { name: "New members join" })).not.toBeInTheDocument();
    await user.click(editor.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(editor.getByRole("button", { name: "Save changes" })).toBeDisabled());
    const request = fetchMock.mock.calls
      .map(([input]) => input as Request)
      .find((request) => request.method === "PATCH")!;
    expect(new URL(request.url).pathname).toBe(endpoint);
    const otherTitle =
      title === "Organization budget groups"
        ? "App × customer organization budget groups"
        : "Organization budget groups";
    const otherTab =
      tab === "organization" ? "Per-App Organization Budget Groups" : "Overall Organization Budget Groups";
    await user.click(screen.getByRole("tab", { name: otherTab }));
    expect(
      within(screen.getByRole("region", { name: otherTitle })).queryByDisplayValue("Starter"),
    ).not.toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: tabTitle }));
    expect(await request.clone().json()).toEqual({
      expected_revision: organization.budget_groups_revision,
      default_budget_template: null,
      default_budgets: [{ name: "Starter", budget_limits: [{ budget_duration: "1mo", max_budget: 0 }] }],
    });
    await user.click(editor.getByRole("button", { name: "Remove Starter" }));
    expect(editor.getByRole("button", { name: "Save changes" })).toBeEnabled();
  });

  it("should save additions, edited limits, the new default and deletions together", async () => {
    const user = userEvent.setup();
    fetchMock.mockImplementation(async () =>
      json({
        ...organization,
        default_budgets: [...organization.default_budgets, { budget_id: "created-id", name: "New" }],
      }),
    );
    render(<CustomerBudgetGroups organization={organization} onSaved={onSaved} />);
    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
    fireEvent.change(screen.getByRole("spinbutton", { name: "Group 1 monthly budget (USD)" }), {
      target: { value: "15" },
    });
    await user.click(screen.getByRole("combobox", { name: "Group 1 at the cap" }));
    await user.click(await screen.findByRole("option", { name: "Pause, allow increase requests" }));
    await user.click(screen.getByRole("button", { name: "Add budget group" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Group name 3" }), { target: { value: "New" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Group 3 monthly budget (USD)" }), {
      target: { value: "30" },
    });
    await user.click(screen.getByRole("combobox", { name: "New members join" }));
    await user.click(await screen.findByRole("option", { name: "New", exact: true }));
    await user.click(screen.getByRole("button", { name: "Remove Old" }));
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
          budget_limits: [{ budget_duration: "1mo", max_budget: 15 }],
        },
        { name: "New", cap_policy: "pause_until_reset", budget_limits: [{ budget_duration: "1mo", max_budget: 30 }] },
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
    await user.click(await screen.findByRole("option", { name: "Standard", exact: true }));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled());
    expect(defaults).toHaveTextContent("Standard");
    await user.click(defaults);
    await user.click(await screen.findByRole("option", { name: "No limit", exact: true }));
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
    render(<CustomerBudgetGroups global={global} organization={source} onSaved={onSaved} />);
    fireEvent.change(screen.getByRole("spinbutton", { name: "Group 1 monthly budget (USD)" }), {
      target: { value: "15" },
    });
    await user.click(screen.getByRole("button", { name: "Remove Old" }));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(error);
    expect(screen.getByRole("spinbutton", { name: "Group 1 monthly budget (USD)" })).toHaveValue(15);
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
    render(
      <CustomerBudgetGroups
        organization={{ ...organization, default_budgets: [], default_budget_template_id: null }}
        onSaved={onSaved}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Add budget group" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Group name 1" }), { target: { value: "First" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Group 1 monthly budget (USD)" }), {
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
    render(
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
