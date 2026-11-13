import userEvent from "@testing-library/user-event";
import { Toaster } from "sonner";
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
  team_count: 0,
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

  it("should save additions, edited limits, the new default and deletions in order", async () => {
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
      ["POST", "/management/v1/customer_organizations/org-1/default_budgets"],
      ["PATCH", "/management/v1/customer_organizations/org-1/default_budgets/standard"],
      ["PATCH", "/management/v1/customer_organizations/org-1"],
      ["DELETE", "/management/v1/customer_organizations/org-1/default_budgets/old"],
    ]);
    expect(await requests[0].clone().json()).toEqual({
      name: "New",
      budget_limits: [{ budget_duration: "1mo", max_budget: 30 }],
    });
    expect(await requests[1].clone().json()).toEqual({ budget_limits: [{ budget_duration: "1mo", max_budget: 15 }] });
    expect(await requests[2].clone().json()).toEqual({ default_budget_template_id: "created-id" });
  });

  it("should refresh after a partial failure and stop before changing the default or deleting", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(
      json({
        ...organization,
        default_budgets: [...organization.default_budgets, { budget_id: "created-id", name: "New" }],
      }),
    );
    fetchMock.mockImplementation(async () => json({ error: { message: "TEMPLATE_IN_USE" } }, 400));
    render(
      <>
        <CustomerBudgetGroups organization={organization} onSaved={onSaved} />
        <Toaster />
      </>,
    );
    fireEvent.change(screen.getByRole("spinbutton", { name: "Group 1 monthly budget (USD)" }), {
      target: { value: "15" },
    });
    await user.click(screen.getByRole("button", { name: "Add budget group" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Group 3 name" }), { target: { value: "New" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Group 3 monthly budget (USD)" }), {
      target: { value: "30" },
    });
    await user.click(screen.getByRole("radio", { name: "Use New as default" }));
    await user.click(screen.getByRole("button", { name: "Remove Old" }));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(await screen.findByText("TEMPLATE_IN_USE")).toBeInTheDocument();
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
