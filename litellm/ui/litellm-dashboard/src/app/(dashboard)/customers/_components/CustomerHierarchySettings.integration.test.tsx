import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { act, chooseSelectOption, fireEvent, render, screen, waitFor, within } from "../../../../../tests/test-utils";
import CustomerHierarchySettings from "./CustomerHierarchySettings";
import type { Organization, Member } from "./customerHierarchyForms";

const audit = {
  created_at: "2026-11-01T00:00:00Z",
  updated_at: "2026-11-01T00:00:00Z",
  created_by: "admin",
  updated_by: "admin",
  spend: 5,
};
const organization: Organization = {
  ...audit,
  customer_org_id: "org-1",
  alias: "Example organization",
  default_budget_id: "template-1",
  default_budgets: [
    { budget_id: "template-1", name: "Standard", is_default: true, max_budget: 10 },
    { budget_id: "template-2", name: "Pro", is_default: false, max_budget: 50 },
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
  parent_team_id: "team-1",
  alias: "Member One",
  budget_id: "template-1",
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

describe("CustomerHierarchySettings", () => {
  beforeEach(() => {
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockReset();
    onSaved.mockReset();
    onDeleted.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

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
    expect(screen.getByRole("spinbutton", { name: "Spend limit (USD)" })).toHaveValue(100);
    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
  });

  it("should preserve other limits when saving a zero organization spend limit", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue(json(organization));
    render(
      <CustomerHierarchySettings
        entity={{ kind: "organization", record: organization }}
        organization={organization}
        onSaved={onSaved}
        onDeleted={onDeleted}
      />,
    );
    act(() =>
      fireEvent.change(screen.getByRole("spinbutton", { name: "Spend limit (USD)" }), { target: { value: "0" } }),
    );
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(lastRequest().method).toBe("PATCH");
    expect(new URL(lastRequest().url).pathname).toBe("/management/v1/customer_organizations/org-1");
    expect(await lastRequest().clone().json()).toEqual({ budget: { ...organization.budget, max_budget: 0 } });
  });

  it("should show friendly template names while sending the selected template identifier", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue(json(member));
    render(
      <CustomerHierarchySettings
        entity={{ kind: "member", record: member }}
        organization={organization}
        onSaved={onSaved}
        onDeleted={onDeleted}
      />,
    );
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
    render(
      <CustomerHierarchySettings
        entity={{ kind: "member", record: member }}
        organization={organization}
        onSaved={onSaved}
        onDeleted={onDeleted}
      />,
    );
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Budget template" }), "No template");
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(await lastRequest().clone().json()).toEqual({ budget_id: null });
  });

  it("should identify an unavailable stored template instead of displaying No template", () => {
    render(
      <CustomerHierarchySettings
        entity={{ kind: "member", record: { ...member, budget_id: "retired-template" } }}
        organization={organization}
        onSaved={onSaved}
        onDeleted={onDeleted}
      />,
    );
    expect(screen.getByRole("combobox", { name: "Budget template" })).toHaveTextContent(
      "Unavailable template (retired-template)",
    );
  });

  it("should clear the member override and its expiry without changing the template", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValue(json(member));
    render(
      <CustomerHierarchySettings
        entity={{ kind: "member", record: member }}
        organization={organization}
        onSaved={onSaved}
        onDeleted={onDeleted}
      />,
    );
    await user.click(screen.getByRole("checkbox", { name: "Override the template budget" }));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(new URL(lastRequest().url).pathname).toBe("/management/v1/customer_members/member-1");
    expect(await lastRequest().clone().json()).toEqual({ budget_override: null, budget_override_expires_at: null });
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

  it("should require confirmation and retain the delete dialog when a parent has children", async () => {
    const user = userEvent.setup();
    fetchMock.mockResolvedValueOnce(
      json({ detail: "Delete the customer's children before deleting the parent." }, 409),
    );
    render(
      <CustomerHierarchySettings
        entity={{ kind: "organization", record: organization }}
        organization={organization}
        onSaved={onSaved}
        onDeleted={onDeleted}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Delete Organization" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete Example organization?" });
    expect(fetchMock).not.toHaveBeenCalled();
    await user.click(within(dialog).getByRole("button", { name: "Delete Organization" }));
    expect(await within(dialog).findByRole("alert")).toHaveTextContent(
      "Delete the customer's children before deleting the parent.",
    );
    expect(dialog).toBeInTheDocument();
    expect(onDeleted).not.toHaveBeenCalled();
    expect(lastRequest().method).toBe("DELETE");
    expect(new URL(lastRequest().url).pathname).toBe("/management/v1/customer_organizations/org-1");
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 204 }));
    await user.click(within(dialog).getByRole("button", { name: "Delete Organization" }));
    await waitFor(() => expect(onDeleted).toHaveBeenCalledOnce());
  });
});
