import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { toast } from "sonner";
import {
  chooseSelectOption,
  fireEvent,
  renderWithProviders,
  screen,
  testQueryClient,
  waitFor,
  within,
} from "../../../../../tests/test-utils";
import CreateCustomerDialog from "./CreateCustomerDialog";

const creations: unknown[] = [];
let rejectCreation = false;
let freeTierReached = false;
let catalogStatus = 200;
let defaultBudgetId: string | null = null;
const onCreated = vi.fn();

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

function Harness() {
  const [open, setOpen] = useState(true);
  return (
    <>
      <button onClick={() => setOpen(true)}>Open customer form</button>
      <CreateCustomerDialog open={open} onOpenChange={setOpen} onCreated={onCreated} />
    </>
  );
}

beforeEach(() => {
  creations.length = 0;
  rejectCreation = false;
  freeTierReached = false;
  catalogStatus = 200;
  defaultBudgetId = null;
  vi.mocked(toast.error).mockClear();
  onCreated.mockClear();
  testQueryClient.clear();
  vi.stubGlobal(
    "fetch",
    vi.fn<typeof fetch>(async (input, init) => {
      const request = new Request(input, init);
      const url = new URL(request.url);
      if (url.pathname === "/models") return Response.json({ data: [] });
      if (url.pathname.endsWith("/.well-known/litellm-ui-config")) {
        return Response.json({ server_root_path: "", proxy_base_url: "", admin_ui_disabled: false });
      }
      if (url.pathname === "/management/v1/customer_organization_budget_groups")
        return Response.json(
          {
            default_budget_template_id: defaultBudgetId,
            default_budgets: [
              { budget_id: "saved-group", name: "100" },
              { budget_id: "custom", name: "Custom" },
              {
                budget_id: "named-group",
                name: "Growth",
                budget_limits: [
                  { max_budget: 1200, budget_duration: "1mo" },
                  { max_budget: 12000, budget_duration: "12mo" },
                ],
                tpm_limit: 50000,
                rpm_limit: 0,
              },
            ],
          },
          { status: catalogStatus },
        );
      if (url.pathname === "/management/v1/customer_organizations" && request.method === "POST") {
        creations.push(await request.json());
        if (freeTierReached)
          return Response.json(
            { detail: "Customer Management free tier limit reached. Upgrade to Enterprise." },
            { status: 403 },
          );
        return rejectCreation
          ? Response.json({ detail: "Customer entity ID already exists." }, { status: 409 })
          : Response.json({ customer_org_id: "created-org", alias: "Example customer" }, { status: 201 });
      }
      throw new Error(`Unexpected request ${request.method} ${url.pathname}`);
    }),
  );
});

afterEach(() => {
  testQueryClient.clear();
  vi.unstubAllGlobals();
});

function renderCustomerForm(name = "Example customer") {
  const user = userEvent.setup();
  renderWithProviders(<Harness />);
  fireEvent.change(screen.getByLabelText("Customer name"), { target: { value: name } });
  return user;
}

describe("CreateCustomerDialog", () => {
  it.each([false, true])("starts with the organization default and respects a custom opt-out (%s)", async (custom) => {
    defaultBudgetId = "named-group";
    const user = renderCustomerForm();
    const profile = screen.getByRole("combobox", { name: "Customer organization limits" });
    await waitFor(() => expect(profile).toHaveTextContent("Growth"));
    if (custom) {
      await user.click(profile);
      await user.click(await screen.findByRole("option", { name: "Enter custom limits instead" }));
    }
    await user.click(screen.getByRole("button", { name: "Create customer" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("created-org"));
    if (custom) expect(creations[0]).toMatchObject({ budget_id: null });
    else expect(creations[0]).not.toHaveProperty("budget_id");
    expect(creations[0]).not.toHaveProperty("budget");
  });

  it.each([
    ["100", "saved-group"],
    ["Growth", "named-group"],
    ["Custom", "custom"],
  ])("creates an organization using the %s profile", async (name, id) => {
    const user = renderCustomerForm();
    const profile = screen.getByRole("combobox", { name: "Customer organization limits" });
    await waitFor(() => expect(profile).toBeEnabled());
    await user.click(profile);
    await user.click(await screen.findByRole("option", { name: new RegExp(`^${name}`) }));
    await user.click(screen.getByRole("button", { name: "Create customer" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledOnce());
    expect(creations[0]).toMatchObject({ budget_id: id });
    expect(creations[0]).not.toHaveProperty("budget");
  });

  it("submits custom limits when the profile catalog is unavailable", async () => {
    catalogStatus = 503;
    const user = renderCustomerForm();
    await chooseSelectOption(
      user,
      screen.getByRole("combobox", { name: "Customer organization limits" }),
      "Enter custom limits instead",
    );
    const limits = within(screen.getByRole("region", { name: "Customer organization limits" }));
    fireEvent.change(limits.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "40" } });
    await userEvent.click(screen.getByRole("button", { name: "Create customer" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledOnce());
    expect(creations[0]).toMatchObject({ budget: { max_budget: 40, budget_duration: "1mo" } });
    expect(creations[0]).not.toHaveProperty("budget_id");
  });

  it("preserves the server default when the catalog fails and limits are untouched", async () => {
    catalogStatus = 503;
    defaultBudgetId = "named-group";
    renderCustomerForm();
    await waitFor(() => expect(screen.getByRole("button", { name: "Create customer" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Create customer" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledOnce());
    expect(creations[0]).not.toHaveProperty("budget_id");
    expect(creations[0]).not.toHaveProperty("budget");
  });

  it("previews profile limits and preserves custom values when switching modes", async () => {
    const user = renderCustomerForm();
    const limits = within(screen.getByRole("region", { name: "Customer organization limits" }));
    const group = within(screen.getByRole("row", { name: "default_user" }));
    fireEvent.change(await limits.findByRole("spinbutton", { name: "Max Budget (USD)" }), {
      target: { value: "15000" },
    });
    fireEvent.change(limits.getByRole("spinbutton", { name: "Tokens per minute" }), { target: { value: "25000" } });
    fireEvent.change(limits.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "120" } });
    fireEvent.change(group.getByRole("spinbutton", { name: "Tokens per minute" }), { target: { value: "5000" } });
    fireEvent.change(group.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "10" } });
    await chooseSelectOption(user, limits.getByRole("combobox", { name: "Reset period" }), "/ year");
    const profile = screen.getByRole("combobox", { name: "Customer organization limits" });
    await waitFor(() => expect(profile).toBeEnabled());
    await user.click(profile);
    await user.click(await screen.findByRole("option", { name: /^Growth/ }));
    expect(profile).toHaveTextContent("Growth");
    expect(profile).toHaveTextContent("$1,200/mo");
    expect(profile).toHaveTextContent("$12,000/yr");
    expect(profile).toHaveTextContent("50K TPM");
    expect(profile).toHaveTextContent("0 RPM");
    expect(limits.queryByRole("spinbutton", { name: "Max Budget (USD)" })).not.toBeInTheDocument();
    expect(limits.queryByRole("spinbutton", { name: "Tokens per minute" })).not.toBeInTheDocument();
    expect(limits.queryByRole("spinbutton", { name: "Requests per minute" })).not.toBeInTheDocument();
    await chooseSelectOption(user, profile, "Enter custom limits instead");
    expect(limits.getByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(15000);
    expect(limits.getByRole("spinbutton", { name: "Tokens per minute" })).toHaveValue(25000);
    expect(limits.getByRole("spinbutton", { name: "Requests per minute" })).toHaveValue(120);
    await user.click(screen.getByRole("button", { name: "Create customer" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledOnce());
    expect(creations[0]).toMatchObject({
      budget: { max_budget: 15000, budget_duration: "12mo", tpm_limit: 25000, rpm_limit: 120 },
      default_budgets: expect.arrayContaining([
        expect.objectContaining({ name: "default_user", tpm_limit: 5000, rpm_limit: 10 }),
      ]),
    });
    expect(creations[0]).not.toHaveProperty("budget_id");
  });

  it("adds budget groups without a secure browser context", async () => {
    vi.stubGlobal("crypto", { getRandomValues: crypto.getRandomValues.bind(crypto), randomUUID: undefined });
    renderWithProviders(<Harness />);
    await userEvent.click(screen.getByRole("button", { name: "Add budget group" }));
    fireEvent.change(screen.getByLabelText("Group name 4"), { target: { value: "extra_group" } });
    await userEvent.click(screen.getByRole("button", { name: "Add budget group" }));
    expect(screen.getByLabelText("Group name 4")).toHaveValue("extra_group");
    expect(screen.getByLabelText("Group name 5")).toHaveValue("");
  });

  it("surfaces the free-tier 403 detail", async () => {
    freeTierReached = true;
    renderCustomerForm("Sixth customer");
    await waitFor(() => expect(screen.getByRole("button", { name: "Create customer" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Create customer" }));
    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith(
        expect.stringContaining("Customer Management free tier limit reached. Upgrade to Enterprise."),
      ),
    );
  });
  it("adds a custom group and selects a remaining default when that group is removed", async () => {
    const user = renderCustomerForm();
    await user.click(screen.getByRole("combobox", { name: "Group 2 at the cap" }));
    await user.click(screen.getByRole("option", { name: "Pause, allow increase requests" }));
    await user.click(screen.getByRole("button", { name: "Add budget group" }));
    fireEvent.change(screen.getByLabelText("Group name 4"), { target: { value: "custom_user" } });
    const group = within(screen.getByRole("row", { name: "custom_user" }));
    fireEvent.change(group.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "300" } });
    await chooseSelectOption(user, group.getByRole("combobox", { name: "Reset period" }), "/ year");
    fireEvent.keyDown(screen.getByRole("combobox", { name: "New members join" }), {
      key: "ArrowDown",
    });
    await user.click(await screen.findByRole("option", { name: "custom_user" }));
    await user.click(screen.getByRole("button", { name: "Remove custom_user" }));
    expect(screen.getByRole("combobox", { name: "New members join" })).toHaveTextContent("power_user");
    await user.click(screen.getByRole("button", { name: "Create customer" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("created-org"));
    expect(creations).toHaveLength(1);
    expect(creations[0]).toMatchObject({
      default_budgets: [
        { name: "power_user", cap_policy: "pause_until_reset" },
        { name: "default_user", cap_policy: "pause_allow_increase" },
        { name: "low_user", cap_policy: "pause_until_reset" },
      ],
      default_budget_template: "power_user",
    });
  });

  it("shows duplicate-name validation without making a request", async () => {
    const user = renderCustomerForm();
    fireEvent.change(screen.getByLabelText("Group name 2"), { target: { value: " power_user " } });
    await user.click(screen.getByRole("button", { name: "Create customer" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Use a unique name for each budget group.");
    expect(creations).toHaveLength(0);
  });

  it("preserves entered budgets after failure and clears the form after a successful retry", async () => {
    rejectCreation = true;
    const user = renderCustomerForm();
    fireEvent.change(screen.getByLabelText("Organization ID (optional)"), { target: { value: "existing-id" } });
    const group = within(screen.getByRole("row", { name: "default_user" }));
    fireEvent.change(group.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "450" } });
    await chooseSelectOption(user, group.getByRole("combobox", { name: "Reset period" }), "/ year");
    await user.click(screen.getByRole("button", { name: "Create customer" }));
    await waitFor(() => expect(creations).toHaveLength(1));
    expect(await screen.findByRole("button", { name: "Create customer" })).toBeEnabled();
    expect(group.getByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(450);
    expect(group.getByRole("combobox", { name: "Reset period" })).toHaveTextContent("/ yr");
    expect(onCreated).not.toHaveBeenCalled();
    rejectCreation = false;
    fireEvent.change(screen.getByLabelText("Organization ID (optional)"), { target: { value: "new-id" } });
    await user.click(screen.getByRole("button", { name: "Create customer" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("created-org"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(creations[1]).toMatchObject({
      customer_org_id: "new-id",
      default_budgets: expect.arrayContaining([
        expect.objectContaining({
          name: "default_user",
          max_budget: 450,
          budget_duration: "12mo",
        }),
      ]),
    });
    await user.click(screen.getByRole("button", { name: "Open customer form" }));
    expect(screen.getByLabelText("Customer name")).toHaveValue("");
    const reopenedGroup = within(screen.getByRole("row", { name: "default_user" }));
    expect(reopenedGroup.getByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(50);
    expect(reopenedGroup.getByRole("combobox", { name: "Reset period" })).toHaveTextContent("/ mo");
    expect(screen.getByRole("combobox", { name: "New members join" })).toHaveTextContent("default_user");
  });
});
