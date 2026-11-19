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
const fetchMock = vi.fn<typeof fetch>();
const onSaved = vi.fn();
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

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
    fetchMock.mockResolvedValue(json(organization));
    render(
      global ? (
        <CustomerBudgetGroups organization={organization} onSaved={onSaved} global />
      ) : (
        <CustomerBudgetGroups organization={organization} onSaved={onSaved} />
      ),
    );
    const rates = within(screen.getByRole("group", { name: "Group 1 rate limits" }));
    fireEvent.change(rates.getByRole("spinbutton", { name: "Tokens per minute" }), { target: { value: "50000" } });
    fireEvent.change(rates.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "0" } });
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
      default_budget_template: "Standard",
      default_budgets: [
        { budget_id: "standard", name: "Standard", tpm_limit: 50000, rpm_limit: 0 },
        { budget_id: "old", name: "Old" },
      ],
    });
  });

  it("creates and removes organization groups through the shared editor in Settings", async () => {
    const user = userEvent.setup();
    const catalog = { ...organization, default_budget_template_id: null, default_budgets: [] };
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      expect(new URL(request.url).pathname).toBe("/management/v1/customer_organization_budget_groups");
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
    renderWithProviders(<CustomerSettings />);
    await user.click(await screen.findByRole("button", { name: "Add budget group" }));
    fireEvent.change(screen.getByLabelText("Group 1 name"), { target: { value: "Starter" } });
    fireEvent.change(screen.getByLabelText("Group 1 monthly budget (USD)"), { target: { value: "0" } });
    expect(screen.queryByRole("radio")).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled());
    const request = fetchMock.mock.calls.at(-1)?.[0] as Request;
    expect(await request.clone().json()).toEqual({
      expected_revision: organization.budget_groups_revision,
      default_budget_template: null,
      default_budgets: [{ name: "Starter", budget_limits: [{ budget_duration: "1mo", max_budget: 0 }] }],
    });
    await user.click(screen.getByRole("button", { name: "Remove Starter" }));
    expect(screen.getByRole("button", { name: "Save changes" })).toBeEnabled();
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
    await user.click(screen.getByRole("option", { name: "Pause, they can request more" }));
    await user.click(screen.getByRole("button", { name: "Add budget group" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Group 3 name" }), { target: { value: "New" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Group 3 monthly budget (USD)" }), {
      target: { value: "30" },
    });
    await user.click(screen.getByRole("radio", { name: "Use New as default" }));
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
    fireEvent.change(screen.getByRole("textbox", { name: "Group 1 name" }), { target: { value: "First" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Group 1 monthly budget (USD)" }), {
      target: { value: "10" },
    });
    expect(screen.getByRole("radio", { name: "Use First as default" })).toBeChecked();
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
