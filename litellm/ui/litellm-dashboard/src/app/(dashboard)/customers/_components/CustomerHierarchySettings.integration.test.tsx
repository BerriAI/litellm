import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { $api } from "@/lib/http/api";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import {
  act,
  chooseSelectOption,
  fireEvent,
  renderWithProviders as render,
  screen,
  testQueryClient,
  waitFor,
  within,
} from "../../../../../tests/test-utils";
import CustomersLayout from "../layout";
import CustomerHierarchySettings from "./CustomerHierarchySettings";
import type { Organization, Member, CustomerHierarchyEntity } from "./customerHierarchyForms";

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: () => ({
    data: { data: [{ id: "openai/gpt-4.1", owned_by: "openai" }] },
    isLoading: false,
    isError: false,
  }),
}));

const audit = {
  created_at: "2026-11-01T00:00:00Z",
  updated_at: "2026-11-01T00:00:00Z",
  created_by: "admin",
  updated_by: "admin",
  spend: 5,
};
const organization: Organization = {
  budget_groups_revision: "a".repeat(64),
  ...audit,
  customer_org_id: "org-1",
  alias: "Example organization",
  default_budget_template_id: "template-1",
  default_budgets: [
    { budget_id: "template-1", name: "Standard", max_budget: 10 },
    { budget_id: "template-2", name: "Pro", max_budget: 50 },
  ],
  budget: {
    max_budget: 100,
    soft_budget: 80,
    budget_duration: "1mo",
    tpm_limit: 200,
    rpm_limit: 10,
    allowed_models: ["openai/gpt-4.1"],
  },
};
const member: Member = {
  ...audit,
  customer_member_id: "member-1",
  parent_team_ids: ["team-1"],
  alias: "Member One",
  budget_id: "template-1",
  budget_override_id: "override-1",
  budget_override: { max_budget: 20 },
  effective_budget: { max_budget: 20 },
  budget_override_expires_at: "2026-12-01T12:00:00Z",
};
const fetchMock = vi.fn<typeof fetch>();
const onSaved = vi.fn();
const onDeleted = vi.fn();
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const lastRequest = (): Request => {
  const request = fetchMock.mock.calls.at(-1)?.[0];
  if (!(request instanceof Request)) throw new Error("Expected a request through the shared HTTP client");
  return request;
};

const renderMemberSettings = (record: Member = member) =>
  render(
    <CustomerHierarchySettings
      entity={{ kind: "member", record }}
      organization={organization}
      onSaved={onSaved}
      onDeleted={onDeleted}
    />,
  );

describe("CustomerHierarchySettings", () => {
  beforeEach(() => {
    registerBaseUrlGetter(() => "http://localhost");
    testQueryClient.clear();
    testQueryClient.setQueryData(
      $api.queryOptions("get", "/management/v1/customer_organization_budget_groups", {}).queryKey,
      { default_budgets: [{ budget_id: "shared", name: "Growth" }] },
    );
    fetchMock.mockReset();
    onSaved.mockReset();
    onDeleted.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    testQueryClient.clear();
    vi.unstubAllGlobals();
  });

  it("sets organization rate limits without requiring a dollar budget", async () => {
    const unlimited = { ...organization, budget: null };
    fetchMock.mockResolvedValue(json(unlimited));
    render(
      <CustomerHierarchySettings
        entity={{ kind: "organization", record: unlimited }}
        organization={unlimited}
        onSaved={onSaved}
        onDeleted={onDeleted}
      />,
    );
    fireEvent.click(screen.getByRole("checkbox", { name: "Set custom limits" }));
    fireEvent.change(screen.getByRole("spinbutton", { name: "Tokens per minute" }), { target: { value: "50000" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "60" } });
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(await lastRequest().clone().json()).toMatchObject({
      budget: { max_budget: null, tpm_limit: 50000, rpm_limit: 60 },
    });
  });

  it("should render the existing organization settings and read-only identifier", () => {
    render(
      <CustomerHierarchySettings
        entity={{ kind: "organization", record: organization }}
        organization={organization}
        onSaved={onSaved}
        onDeleted={onDeleted}
      />,
    );
    expect(screen.getByRole("textbox", { name: "Display name" })).toHaveValue("Example organization");
    expect(screen.getByRole("textbox", { name: "Identifier" })).toHaveAttribute("readonly");
    expect(screen.getByRole("combobox", { name: "Customer budget (USD)" })).toHaveValue("100");
    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
  });

  it.each([
    ["1mo", false, "$0"],
    ["7d", false, "$0"],
    [null, false, "$0"],
    ["1mo", true, "$0"],
    ["7d", true, "$0"],
    [null, true, "$0"],
    ["7d", false, ""],
    ["7d", true, ""],
    ["7d", false, "$0", null],
    ["7d", true, "", null],
  ] as const)("keeps reset %s via group %s with %s", async (budget_duration, viaGroup, amount, max_budget = 100) => {
    const user = userEvent.setup();
    const custom = {
      ...organization,
      budget_id: "owned",
      budget: {
        ...organization.budget,
        max_budget,
        budget_duration,
        budget_limits: [{ max_budget: 1000, budget_duration: "12mo" }],
      },
    };
    fetchMock.mockResolvedValue(json(custom));
    render(
      <CustomerHierarchySettings
        entity={{ kind: "organization", record: custom }}
        organization={custom}
        onSaved={onSaved}
        onDeleted={onDeleted}
      />,
    );
    const save = screen.getByRole("button", { name: "Save changes" });
    const budget = screen.getByRole("combobox", { name: "Customer budget (USD)" });
    fireEvent.change(screen.getByRole("textbox", { name: "Display name" }), { target: { value: "Renamed" } });
    await user.clear(budget);
    if (viaGroup) {
      await user.click(budget);
      await user.click(screen.getByRole("option", { name: "Growth", exact: true }));
      await user.clear(budget);
    }
    if (amount) await user.type(budget, amount);
    await user.click(save);
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(lastRequest().method).toBe("PATCH");
    expect(new URL(lastRequest().url).pathname).toBe("/management/v1/customer_organizations/org-1");
    expect(await lastRequest().clone().json()).toEqual({
      alias: "Renamed",
      ...(amount || max_budget != null ? { budget: { ...custom.budget, max_budget: amount ? 0 : null } } : {}),
    });
  });

  it.each(["$175", "", "Growth"])("changes a group selection to %s", async (amount) => {
    const user = userEvent.setup();
    const grouped = { ...organization, budget_id: "shared" };
    fetchMock.mockImplementation(async () => json(grouped));
    render(
      <CustomerHierarchySettings
        entity={{ kind: "organization", record: grouped }}
        organization={grouped}
        onSaved={onSaved}
        onDeleted={onDeleted}
      />,
    );
    const save = screen.getByRole("button", { name: "Save changes" });
    const budget = screen.getByRole("combobox", { name: "Customer budget (USD)" });
    expect(budget).toHaveValue("Growth");
    await user.clear(budget);
    if (amount) await user.type(budget, amount === "Growth" ? "$175" : amount);
    if (amount === "Growth") {
      fireEvent.change(screen.getByRole("spinbutton", { name: "Tokens per minute", hidden: true }), {
        target: { value: "300" },
      });
      await user.clear(budget);
      await user.click(screen.getByRole("option", { name: "Growth", exact: true }));
      expect(save).toBeDisabled();
      fireEvent.change(screen.getByRole("textbox", { name: "Display name" }), { target: { value: "Renamed" } });
    }
    await user.click(save);
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(await lastRequest().clone().json()).toEqual(
      amount === "Growth"
        ? { alias: "Renamed" }
        : { budget: { ...organization.budget, max_budget: amount ? 175 : null, budget_limits: null } },
    );
  });

  it.each([
    [true, false],
    [true, true],
    [false, false],
    [false, true],
  ] as const)("protects shared=%s budget while catalog unavailable=%s", async (shared, unavailable) => {
    testQueryClient.clear();
    const grouped = { ...organization, budget_id: shared ? "shared" : "owned", budget_is_shared: shared };
    fetchMock.mockImplementation(async (request) => {
      if (request instanceof Request && request.url.endsWith("customer_organization_budget_groups"))
        return unavailable ? json({ detail: "Unavailable" }, 503) : new Promise<Response>(() => {});
      return json(grouped);
    });
    render(
      <CustomerHierarchySettings
        entity={{ kind: "organization", record: grouped }}
        organization={grouped}
        onSaved={onSaved}
        onDeleted={onDeleted}
      />,
    );
    const budget = screen.getByRole("combobox", { name: "Customer budget (USD)" });
    if (unavailable) await screen.findByRole("button", { name: "Retry" });
    if (shared) {
      expect(budget).toBeDisabled();
      expect(screen.queryByRole("spinbutton", { name: "Tokens per minute" })).not.toBeInTheDocument();
    } else {
      expect(budget).toBeEnabled();
      await userEvent.clear(budget);
      await userEvent.type(budget, "$40");
      await userEvent.keyboard("{Escape}");
      fireEvent.change(screen.getByRole("spinbutton", { name: "Tokens per minute", hidden: true }), {
        target: { value: "300" },
      });
    }
    fireEvent.change(screen.getByRole("textbox", { name: "Display name" }), { target: { value: "Renamed" } });
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(await lastRequest().clone().json()).toEqual({
      alias: "Renamed",
      ...(!shared ? { budget: { ...organization.budget, max_budget: 40, tpm_limit: 300 } } : {}),
    });
  });

  it.each([false, true])("toggles custom assignment from enabled %s", async (enabled) => {
    const custom = {
      ...organization,
      budget_id: enabled ? "owned" : null,
      budget: enabled ? organization.budget : null,
    };
    fetchMock.mockResolvedValue(json(custom));
    render(
      <CustomerHierarchySettings
        entity={{ kind: "organization", record: custom }}
        organization={custom}
        onSaved={onSaved}
        onDeleted={onDeleted}
      />,
    );
    await userEvent.click(screen.getByRole("checkbox", { name: "Set custom limits" }));
    expect(screen.getByRole("combobox", { name: "Customer budget (USD)" })).toHaveValue("");
    if (!enabled) {
      fireEvent.change(screen.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "10" } });
      const budget = screen.getByRole("combobox", { name: "Customer budget (USD)" });
      await userEvent.type(budget, "$40");
      await userEvent.clear(budget);
      await userEvent.keyboard("{Escape}");
    }
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(await lastRequest().clone().json()).toEqual({
      budget: enabled
        ? null
        : {
            max_budget: null,
            soft_budget: null,
            tpm_limit: null,
            rpm_limit: 10,
            budget_duration: null,
            allowed_models: null,
          },
    });
  });

  it("should show friendly template names while sending the selected template identifier", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue(json(member));
    renderMemberSettings();
    const template = screen.getByRole("combobox", { name: "Budget template" });
    expect(template).toHaveTextContent("Standard (default)");
    expect(template).not.toHaveTextContent("template-1");
    await chooseSelectOption(user, template, "Pro");
    expect(template).toHaveTextContent("Pro");
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(new URL(lastRequest().url).pathname).toBe("/management/v1/customer_members/member-1");
    expect(await lastRequest().clone().json()).toEqual({ budget_id: "template-2" });
  });

  it("should clear the template without changing the existing override", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue(json({ ...member, budget_id: null }));
    renderMemberSettings();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Budget template" }), "No template");
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(await lastRequest().clone().json()).toEqual({ budget_id: null });
  });

  it("should identify an unavailable stored template instead of displaying No template", () => {
    renderMemberSettings({ ...member, budget_id: "retired-template" });
    expect(screen.getByRole("combobox", { name: "Budget template" })).toHaveTextContent(
      "Unavailable template (retired-template)",
    );
  });

  it("should clear the member override and its expiry without changing the template", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue(json(member));
    renderMemberSettings();
    await user.click(screen.getByRole("checkbox", { name: "Override the template budget" }));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(new URL(lastRequest().url).pathname).toBe("/management/v1/customer_members/member-1");
    expect(await lastRequest().clone().json()).toEqual({ budget_override: null, budget_override_expires_at: null });
  });

  it("should make the entered limits permanent when the administrator clears Override expires", async () => {
    fetchMock.mockResolvedValue(json(member));
    renderMemberSettings();
    expect(screen.getByText("Uses your local time. Leave blank for no expiry.")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Override expires"), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(await lastRequest().clone().json()).toEqual({
      budget: { max_budget: 20 },
      budget_override: null,
      budget_override_expires_at: null,
    });
  });

  it("should require a preset before disabling a private permanent member budget", async () => {
    const user = userEvent.setup();
    const privateMember: Member = {
      ...member,
      budget_id: "private-1",
      budget: { max_budget: 750 },
      budget_override_id: null,
      budget_override: null,
      budget_override_expires_at: null,
    };
    fetchMock.mockResolvedValue(json({ ...privateMember, budget_id: "template-2" }));
    renderMemberSettings(privateMember);
    const template = screen.getByRole("combobox", { name: "Budget template" });
    expect(template).toHaveTextContent("No template");
    await user.click(screen.getByRole("checkbox", { name: "Override the template budget" }));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Select a budget template or enable an override for this customer member.",
    );
    expect(fetchMock).not.toHaveBeenCalled();
    await chooseSelectOption(user, template, "Pro");
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(await lastRequest().clone().json()).toEqual({ budget_id: "template-2" });
  });

  it("should explain that blank member limits inherit from the selected template", () => {
    renderMemberSettings();
    expect(
      screen.getByText("Blank limits inherit from the budget template. Without a template, blank means no limit."),
    ).toBeInTheDocument();
    expect(screen.queryByText("Leave a limit blank for no limit. Zero is a valid limit.")).not.toBeInTheDocument();
  });

  it("should reject combining a new preset with zero permanent custom limits", async () => {
    const user = userEvent.setup();
    const inherited = { ...member, budget_override: null, budget_override_expires_at: null };
    renderMemberSettings(inherited);
    await user.click(screen.getByRole("checkbox", { name: "Override the template budget" }));
    fireEvent.change(screen.getByRole("spinbutton", { name: "Spend limit (USD)" }), { target: { value: "0" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "0" } });
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Budget template" }), "Pro");
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Save the selected preset first, then set custom limits.");
    expect(fetchMock).not.toHaveBeenCalled();
    expect(onSaved).not.toHaveBeenCalled();
  });

  it("should omit untouched preset restrictions when saving the first custom member budget", async () => {
    const user = userEvent.setup();
    const inheritedMember = {
      ...member,
      budget_override: null,
      effective_budget: { max_budget: 100, rpm_limit: 10, allowed_models: ["openai/gpt-4.1"] },
    };
    fetchMock.mockResolvedValue(json(inheritedMember));
    renderMemberSettings(inheritedMember);
    await user.click(screen.getByRole("checkbox", { name: "Override the template budget" }));
    expect(screen.getByRole("spinbutton", { name: "Spend limit (USD)" })).toHaveValue(null);
    expect(screen.getByRole("spinbutton", { name: "Requests per minute" })).toHaveValue(null);
    act(() =>
      fireEvent.change(screen.getByRole("spinbutton", { name: "Spend limit (USD)" }), { target: { value: "50" } }),
    );
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(await lastRequest().clone().json()).toEqual({
      budget: { max_budget: 50 },
    });
  });

  it("should keep edited values and show the API error when a save fails", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue(json({ detail: "Database not connected." }, 503));
    render(
      <CustomerHierarchySettings
        entity={{ kind: "organization", record: organization }}
        organization={organization}
        onSaved={onSaved}
        onDeleted={onDeleted}
      />,
    );
    const alias = screen.getByRole("textbox", { name: "Display name" });
    await user.clear(alias);
    await user.type(alias, "Updated organization");
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Database not connected.");
    expect(alias).toHaveValue("Updated organization");
    expect(onSaved).not.toHaveBeenCalled();
  });

  it("should require the exact record identifier before enabling deletion", async () => {
    const user = userEvent.setup();
    render(
      <CustomerHierarchySettings
        entity={{ kind: "organization", record: organization }}
        organization={organization}
        onSaved={onSaved}
        onDeleted={onDeleted}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Delete customer organization" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete customer organization" });
    const button = within(dialog).getByRole("button", { name: "Delete" });
    expect(button).toBeDisabled();
    expect(dialog).toHaveTextContent("Example organization");
    expect(dialog).toHaveTextContent("org-1");
    const confirmation = within(dialog).getByPlaceholderText("org-1");
    await user.type(confirmation, "org");
    expect(button).toBeDisabled();
    await user.clear(confirmation);
    await user.type(confirmation, "org-1");
    expect(button).toBeEnabled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([
    [401, "Only proxy admins can delete this record."],
    [409, "The delete transaction could not complete."],
  ] as const)(
    "should retain the dialog and show a %s server error before a successful retry",
    async (status, message) => {
      const user = userEvent.setup();
      fetchMock.mockResolvedValueOnce(json({ detail: message }, status));
      render(
        <CustomerHierarchySettings
          entity={{ kind: "organization", record: organization }}
          organization={organization}
          onSaved={onSaved}
          onDeleted={onDeleted}
        />,
      );
      await user.click(screen.getByRole("button", { name: "Delete customer organization" }));
      const dialog = await screen.findByRole("dialog", { name: "Delete customer organization" });
      await user.type(within(dialog).getByPlaceholderText("org-1"), "org-1");
      await user.click(within(dialog).getByRole("button", { name: "Delete" }));
      expect(await within(dialog).findByText(message)).toHaveAttribute("role", "alert");
      expect(dialog).toBeInTheDocument();
      expect(onDeleted).not.toHaveBeenCalled();
      expect(lastRequest().method).toBe("DELETE");
      const url = new URL(lastRequest().url);
      expect(url.pathname).toBe("/management/v1/customer_organizations/org-1");
      expect(url.searchParams.get("cascade")).toBe("true");
      fetchMock.mockResolvedValueOnce(new Response(null, { status: 204 }));
      await user.click(within(dialog).getByRole("button", { name: "Delete" }));
      await waitFor(() => expect(onDeleted).toHaveBeenCalledOnce());
    },
  );

  it.each([
    { kind: "team", record: { ...audit, customer_team_id: "team-1", parent_org_id: "org-1", alias: "Support" } },
    { kind: "member", record: member },
  ] satisfies CustomerHierarchyEntity[])(
    "should delete the selected customer $kind with the correct cascade policy",
    async (entity) => {
      const user = userEvent.setup();
      fetchMock.mockResolvedValueOnce(new Response(null, { status: 204 }));
      render(
        <CustomerHierarchySettings
          entity={entity}
          organization={organization}
          onSaved={onSaved}
          onDeleted={onDeleted}
        />,
      );
      await user.click(screen.getByRole("button", { name: `Delete customer ${entity.kind}` }));
      const dialog = await screen.findByRole("dialog", { name: `Delete customer ${entity.kind}` });
      const id = entity.kind === "team" ? entity.record.customer_team_id : entity.record.customer_member_id;
      await user.type(within(dialog).getByPlaceholderText(id), id);
      await user.click(within(dialog).getByRole("button", { name: "Delete" }));
      await waitFor(() => expect(onDeleted).toHaveBeenCalledOnce());
      const url = new URL(lastRequest().url);
      expect(url.pathname).toBe(`/management/v1/customer_${entity.kind === "team" ? "teams" : "members"}/${id}`);
      expect(url.searchParams.get("cascade")).toBe(entity.kind === "team" ? "true" : null);
    },
  );

  it("clears the free-tier warning when a successful deletion frees capacity", async () => {
    let atCapacity = true;
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      if (request.method === "DELETE") {
        atCapacity = false;
        return new Response(null, { status: 204 });
      }
      const limits = {
        limits_apply: true,
        organization_limit: 5,
        teams_per_organization_limit: 5,
        members_per_team_limit: 5,
        organization_count: 1,
        any_limit_reached: atCapacity,
      };
      return json(limits);
    });
    render(
      <CustomersLayout>
        <CustomerHierarchySettings
          entity={{ kind: "member", record: member }}
          organization={organization}
          onSaved={onSaved}
          onDeleted={onDeleted}
        />
      </CustomersLayout>,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent("Customer Management free tier limit reached");
    fireEvent.click(screen.getByRole("button", { name: "Delete customer member" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete customer member" });
    fireEvent.change(within(dialog).getByPlaceholderText("member-1"), { target: { value: "member-1" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    expect(screen.getByText(/Free tier: up to 5 customers/)).toBeInTheDocument();
  });

  it("should keep the confirmation pending until a delete returns204", async () => {
    const user = userEvent.setup();
    let settle: ((response: Response) => void) | undefined;
    fetchMock.mockReturnValueOnce(
      new Promise<Response>((resolve) => {
        settle = resolve;
      }),
    );
    renderMemberSettings();
    await user.click(screen.getByRole("button", { name: "Delete customer member" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete customer member" });
    await user.type(within(dialog).getByPlaceholderText("member-1"), "member-1");
    await user.click(within(dialog).getByRole("button", { name: "Delete" }));
    expect(within(dialog).getByRole("button", { name: "Deleting..." })).toBeDisabled();
    expect(within(dialog).getByRole("button", { name: "Cancel" })).toBeDisabled();
    await user.keyboard("{Escape}");
    expect(dialog).toBeInTheDocument();
    expect(onDeleted).not.toHaveBeenCalled();
    act(() => settle?.(new Response(null, { status: 204 })));
    await waitFor(() => expect(onDeleted).toHaveBeenCalledOnce());
  });
});
