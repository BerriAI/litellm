import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { act, fireEvent, renderWithProviders as render, screen, waitFor } from "../../../../../tests/test-utils";
import CreateCustomerHierarchyDialog from "./CreateCustomerHierarchyDialog";
import type { Organization, Team } from "./customerHierarchyForms";

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: () => ({ data: { data: [] }, isLoading: false, isError: false }),
}));

const audit = {
  created_at: "2026-11-01T00:00:00Z",
  updated_at: "2026-11-01T00:00:00Z",
  created_by: "admin",
  updated_by: "admin",
  spend: 0,
};
const organization: Organization = {
  budget_groups_revision: "a".repeat(64),
  ...audit,
  customer_org_id: "org-1",
  alias: "Example customer",
  default_budget_template_id: "standard-template",
  default_budgets: [{ name: "Standard", is_default: true, budget_id: "standard-template", max_budget: 100 }],
};
const team: Team = { ...audit, customer_team_id: "team-1", parent_org_id: "org-1", alias: "Support" };
const fetchMock = vi.fn<typeof fetch>();
const onOpenChange = vi.fn();
const onCreated = vi.fn();
const json = (body: unknown, status = 201) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const requestBody = async () => {
  const request = fetchMock.mock.calls[0]?.[0];
  if (!(request instanceof Request)) throw new Error("Expected shared-client request");
  return { method: request.method, path: new URL(request.url).pathname, body: await request.clone().json() };
};

describe("CreateCustomerHierarchyDialog", () => {
  beforeEach(() => {
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockReset();
    onOpenChange.mockReset();
    onCreated.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  it("should render customer team creation under its organization", () => {
    render(
      <CreateCustomerHierarchyDialog
        kind="team"
        organization={organization}
        open
        onOpenChange={onOpenChange}
        onCreated={onCreated}
      />,
    );
    expect(screen.getByRole("dialog", { name: "Create customer team" })).toHaveTextContent("Example customer");
    expect(screen.getByRole("textbox", { name: "Display name" })).toHaveValue("");
  });

  it("should create a customer team with a zero spend limit and wait for201 before closing", async () => {
    const user = userEvent.setup();
    let settle: ((response: Response) => void) | undefined;
    fetchMock.mockReturnValue(
      new Promise<Response>((resolve) => {
        settle = resolve;
      }),
    );
    render(
      <CreateCustomerHierarchyDialog
        kind="team"
        organization={organization}
        open
        onOpenChange={onOpenChange}
        onCreated={onCreated}
      />,
    );
    await user.type(screen.getByRole("textbox", { name: "Display name" }), "Engineering");
    await user.click(screen.getByRole("checkbox", { name: "Set a budget and limits" }));
    act(() =>
      fireEvent.change(screen.getByRole("spinbutton", { name: "Spend limit (USD)" }), { target: { value: "0" } }),
    );
    await user.click(screen.getByRole("button", { name: "Create" }));
    expect(screen.getByRole("button", { name: "Creating…" })).toBeDisabled();
    await user.keyboard("{Escape}");
    expect(onCreated).not.toHaveBeenCalled();
    expect(onOpenChange).not.toHaveBeenCalled();
    expect(await requestBody()).toMatchObject({
      method: "POST",
      path: "/management/v1/customer_teams",
      body: { parent_org_id: "org-1", alias: "Engineering", budget: { max_budget: 0 } },
    });
    act(() => settle?.(json({ ...team, customer_team_id: "engineering", alias: "Engineering" })));
    await waitFor(() =>
      expect(onCreated).toHaveBeenCalledWith({
        kind: "team",
        record: { ...team, customer_team_id: "engineering", alias: "Engineering" },
      }),
    );
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("should create a customer member with its preselected default template", async () => {
    const user = userEvent.setup();
    const created = {
      ...audit,
      customer_member_id: "new-member",
      parent_team_ids: ["team-1"],
      alias: "Jamie",
      budget_id: "standard-template",
      budget_override_id: null,
      effective_budget: { max_budget: 100 },
    };
    fetchMock.mockResolvedValue(json(created));
    render(
      <CreateCustomerHierarchyDialog
        kind="member"
        organization={organization}
        team={team}
        open
        onOpenChange={onOpenChange}
        onCreated={onCreated}
      />,
    );
    expect(screen.getByRole("combobox", { name: "Budget template" })).toHaveTextContent("Standard (default)");
    await user.click(screen.getByRole("combobox", { name: "Budget template" }));
    expect(screen.queryByRole("option", { name: "No template" })).not.toBeInTheDocument();
    await user.keyboard("{Escape}");
    await user.type(screen.getByRole("textbox", { name: "Display name" }), "Jamie");
    await user.click(screen.getByRole("button", { name: "Create" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith({ kind: "member", record: created }));
    expect(await requestBody()).toMatchObject({
      method: "POST",
      path: "/management/v1/customer_members",
      body: {
        parent_team_ids: ["team-1"],
        alias: "Jamie",
        budget_id: "standard-template",
        budget_override: null,
      },
    });
  });

  it("should retain the form values and keep the dialog open when creation fails", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue(json({ detail: "Customer entity ID already exists." }, 409));
    render(
      <CreateCustomerHierarchyDialog
        kind="team"
        organization={organization}
        open
        onOpenChange={onOpenChange}
        onCreated={onCreated}
      />,
    );
    await user.type(screen.getByRole("textbox", { name: "Display name" }), "Engineering");
    await user.click(screen.getByRole("button", { name: "Create" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Customer entity ID already exists.");
    expect(screen.getByRole("textbox", { name: "Display name" })).toHaveValue("Engineering");
    expect(onOpenChange).not.toHaveBeenCalled();
    expect(onCreated).not.toHaveBeenCalled();
  });

  it.each(["team", "member"] as const)("surfaces the free-tier 403 detail when creating a %s", async (kind) => {
    fetchMock.mockResolvedValue(json({ detail: "Free customer portal limit reached. Upgrade to Enterprise." }, 403));
    render(
      <CreateCustomerHierarchyDialog
        kind={kind}
        organization={organization}
        team={team}
        open
        onOpenChange={onOpenChange}
        onCreated={onCreated}
      />,
    );
    fireEvent.change(screen.getByRole("textbox", { name: "Display name" }), { target: { value: "Example" } });
    fireEvent.click(screen.getByRole("button", { name: "Create" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Free customer portal limit reached. Upgrade to Enterprise.",
    );
    expect(onOpenChange).not.toHaveBeenCalled();
    expect(onCreated).not.toHaveBeenCalled();
  });
});
