import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { render, screen, fireEvent, waitFor } from "../../../../../tests/test-utils";
import CustomerBudgetGroups from "./CustomerBudgetGroups";
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
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockReset();
    onSaved.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

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

  it("should keep the whole draft after a rejected save and allow retry", async () => {
    const user = userEvent.setup();
    fetchMock.mockImplementation(async () => json({ detail: "A referenced template cannot be deleted." }, 409));
    render(<CustomerBudgetGroups organization={organization} onSaved={onSaved} />);
    fireEvent.change(screen.getByRole("spinbutton", { name: "Group 1 monthly budget (USD)" }), {
      target: { value: "15" },
    });
    await user.click(screen.getByRole("button", { name: "Remove Old" }));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("A referenced template cannot be deleted.");
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
