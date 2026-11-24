import userEvent from "@testing-library/user-event";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import {
  jsonResponse as json,
  chooseSelectOption,
  fireEvent,
  renderWithProviders as render,
  screen,
  testQueryClient,
  waitFor,
  within,
} from "../../../../../tests/test-utils";
import CustomerBudgetSummary from "./CustomerBudgetSummary";
import { customerUsageEntity } from "./customerUsageApi";
import type { Member, Organization } from "./customerHierarchyForms";

const audit = { created_at: "", updated_at: "", created_by: "", updated_by: "", spend: 10 };
const standard = { budget_id: "standard", name: "Standard", max_budget: 500, allowed_models: ["preset-model"] };
const organization: Organization = {
  customer_org_id: "org-1",
  budget_groups_revision: "a".repeat(64),
  member_count: 1,
  alias: "Example",
  ...audit,
  default_budgets: [],
  budget_id: "owned",
  budget_is_shared: false,
  budget: {
    max_budget: 100,
    soft_budget: 80,
    budget_duration: "7d",
    tpm_limit: 1000,
    rpm_limit: 10,
    allowed_models: ["configured-model"],
    budget_limits: [{ max_budget: 1000, budget_duration: "12mo", reset_at: "2027-01-01T00:00:00Z" }],
  },
};
const memberOrganization: Organization = {
  ...audit,
  customer_org_id: "org-1",
  budget_groups_revision: organization.budget_groups_revision,
  member_count: 1,
  default_budgets: [standard],
};
const baseBudget = {
  max_budget: 100,
  budget_duration: "1mo",
  soft_budget: 80,
  tpm_limit: 1000,
  rpm_limit: 10,
  allowed_models: ["configured-model"],
  budget_limits: [
    { max_budget: 1000, budget_duration: "12mo", reset_at: "2027-01-01T00:00:00Z" },
    { max_budget: 30, budget_duration: "1mo" },
  ],
} satisfies NonNullable<Member["budget"]>;
const member: Member = {
  ...audit,
  customer_member_id: "member-1",
  parent_org_id: "org-1",
  parent_team_ids: ["team-1"],
  alias: "Example member",
  budget_id: "owned",
  budget: baseBudget,
  budget_override: { max_budget: 1000, budget_duration: "1mo", rpm_limit: 100 },
  budget_override_expires_at: "2027-01-02T04:05:06.000Z",
  effective_budget: { ...baseBudget, max_budget: 1000, rpm_limit: 100 },
};
const fetchMock = vi.fn<typeof fetch>();
const writes = () =>
  fetchMock.mock.calls
    .map(([input]) => input)
    .filter((input): input is Request => input instanceof Request && input.method === "PATCH");
const views = {
  organization: {
    entity: {
      id: "org-1",
      label: "Example",
      spend: 10,
      budgetLimit: 100,
      budgetDuration: "7d",
      budgetLimits: [],
      budgetResetAt: null,
    },
    memberId: null,
    dialog: "Edit organization budget",
    error: "Budget update unavailable",
  },
  member: {
    entity: customerUsageEntity(member.customer_member_id, member),
    memberId: member.customer_member_id,
    dialog: "Edit member budget",
    error: "Member budget unavailable",
  },
};

let viewScope: keyof typeof views;

async function openBudget(scope: keyof typeof views) {
  viewScope = scope;
  const { entity, memberId } = views[scope];
  const user = userEvent.setup();
  render(
    <CustomerBudgetSummary
      entity={entity}
      scope={{ org: "org-1", team: null, member: memberId }}
      settingsHref={`/customers/manage?org=org-1${memberId ? `&member=${memberId}` : ""}&tab=settings`}
    />,
  );
  await user.click(screen.getByRole("button", { name: "Edit budget" }));
  return user;
}

beforeEach(() => {
  testQueryClient.clear();
  registerBaseUrlGetter(() => "http://localhost");
  fetchMock.mockReset();
  fetchMock.mockImplementation(async (input) => {
    const path = new URL((input as Request).url).pathname;
    if (path.endsWith("/customer_organization_budget_groups")) return json({ default_budgets: [standard] });
    if (path.endsWith("/customer_organizations/org-1"))
      return json(viewScope === "member" ? memberOrganization : organization);
    if (path.endsWith("/customer_members/member-1")) return json(member);
    if (path.endsWith("/models")) return json({ data: [{ id: "configured-model", owned_by: "openai" }] });
    throw new Error(`Unexpected request ${path}`);
  });
  vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => fetchMock(new Request(input, init)));
});
afterEach(() => vi.unstubAllGlobals());

it.each(["organization", "member"] as const)(
  "replaces removed %s spend rows and ignores a blank new row",
  async (scope) => {
    const user = await openBudget(scope);
    const dialog = within(await screen.findByRole("dialog", { name: views[scope].dialog }));
    for (const remove of (await dialog.findAllByRole("button", { name: "Remove limit" })).toReversed())
      await user.click(remove);
    expect(dialog.queryByText("Additional Max Budget (USD)")).not.toBeInTheDocument();
    const add = dialog.getByRole("button", { name: "Add another Max Budget" });
    await user.click(add);
    fireEvent.change(dialog.getByRole("spinbutton", { name: "Additional Max Budget (USD)" }), {
      target: { value: "5000" },
    });
    await user.click(add);
    await user.click(dialog.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(writes()).toHaveLength(1));
    expect(await writes()[0].clone().json()).toEqual({
      budget: { budget_limits: [{ max_budget: 5000, budget_duration: scope === "organization" ? "1mo" : "12mo" }] },
    });
  },
);

describe("organization budget dialog", () => {
  it("opens from the budget summary and saves custom spend, soft budget, rates and reset period", async () => {
    const user = await openBudget("organization");
    const dialog = within(await screen.findByRole("dialog", { name: "Edit organization budget" }));
    expect(await dialog.findByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(100);
    expect(dialog.getByRole("textbox", { name: "Custom period" })).toHaveValue("7d");
    fireEvent.change(dialog.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "200" } });
    fireEvent.change(dialog.getByRole("spinbutton", { name: "Soft budget (USD)" }), { target: { value: "150" } });
    fireEvent.change(dialog.getByRole("spinbutton", { name: "Tokens per minute" }), { target: { value: "2000" } });
    fireEvent.change(dialog.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "0" } });
    await user.click(dialog.getAllByRole("combobox", { name: "Reset period" })[0]);
    expect((await screen.findAllByRole("option")).map((option) => option.textContent)).toEqual([
      "/ month",
      "/ year",
      "No reset",
      "Custom period",
    ]);
    await user.click(screen.getByRole("option", { name: "/ month" }));
    await user.click(dialog.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(writes()).toHaveLength(1);
    expect(new URL(writes()[0].url).pathname).toBe("/management/v1/customer_organizations/org-1");
    expect(await writes()[0].clone().json()).toEqual({
      budget: {
        max_budget: 200,
        soft_budget: 150,
        budget_duration: "1mo",
        tpm_limit: 2000,
        rpm_limit: 0,
      },
    });
  });

  it("selects an organization preset and submits its ID without custom overrides", async () => {
    const user = await openBudget("organization");
    const source = await screen.findByRole("combobox", { name: "Customer organization limits" });
    await waitFor(() => expect(source).toBeEnabled());
    await chooseSelectOption(user, source, /^Standard/);
    expect(source).toHaveTextContent("Allowed models: preset-model");
    expect(screen.queryByRole("spinbutton", { name: "Max Budget (USD)" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(writes()).toHaveLength(1));
    expect(await writes()[0].clone().json()).toEqual({ budget_id: "standard" });
  });

  it.each(["9000", ""])("restores the complete saved budget after discarding a window edit %s", async (amount) => {
    const user = await openBudget("organization");
    const source = await screen.findByRole("combobox", { name: "Customer organization limits" });
    await waitFor(() => expect(source).toBeEnabled());
    fireEvent.change(screen.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "200" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Additional Max Budget (USD)" }), {
      target: { value: amount },
    });
    await chooseSelectOption(user, source, /^Standard/);
    await chooseSelectOption(user, source, "Enter custom limits instead");
    expect(screen.getByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(100);
    expect(screen.getByRole("spinbutton", { name: "Additional Max Budget (USD)" })).toHaveValue(1000);
    fireEvent.change(screen.getByRole("spinbutton", { name: "Additional Max Budget (USD)" }), {
      target: { value: "1200" },
    });
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(writes()).toHaveLength(1));
    expect(await writes()[0].clone().json()).toEqual({
      budget: { budget_limits: [{ ...organization.budget!.budget_limits![0], max_budget: 1200 }] },
    });
  });
});

describe("member budget dialog", () => {
  it("saves monthly and yearly base limits without changing the active temporary grant", async () => {
    const user = await openBudget("member");
    const dialog = within(await screen.findByRole("dialog", { name: "Edit member budget" }));
    expect(await dialog.findByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(100);
    fireEvent.change(dialog.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "200" } });
    fireEvent.change(dialog.getAllByRole("spinbutton", { name: "Additional Max Budget (USD)" })[0], {
      target: { value: "2400" },
    });
    fireEvent.change(dialog.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "0" } });
    await user.click(dialog.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(writes()).toHaveLength(1);
    expect(new URL(writes()[0].url).pathname).toBe("/management/v1/customer_members/member-1");
    expect(await writes()[0].clone().json()).toEqual({
      budget: {
        budget_duration: "1mo",
        max_budget: 200,
        rpm_limit: 0,
        budget_limits: [{ ...baseBudget.budget_limits[0], max_budget: 2400 }, baseBudget.budget_limits[1]],
      },
    });
  });

  it("selects a preset without removing the temporary grant or its expiry", async () => {
    const user = await openBudget("member");
    const source = await screen.findByRole("combobox", { name: "Member limits" });
    await chooseSelectOption(user, source, /^Standard/);
    expect(screen.queryByRole("spinbutton", { name: "Max Budget (USD)" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(writes()).toHaveLength(1));
    expect(await writes()[0].clone().json()).toEqual({
      budget_id: "standard",
    });
  });

  it("replaces a preset monthly cap with a permanent custom cap", async () => {
    const presetMember = { ...member, budget_id: "standard", budget: { max_budget: 500, budget_duration: "1mo" } };
    const serve = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (input, init) =>
      new URL((input as Request).url).pathname.endsWith("/customer_members/member-1")
        ? json(presetMember)
        : serve(input, init),
    );
    const user = await openBudget("member");
    await chooseSelectOption(
      user,
      await screen.findByRole("combobox", { name: "Member limits" }),
      "Enter custom limits instead",
    );
    expect(screen.getByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(500);
    fireEvent.change(screen.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "750" } });
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(writes()).toHaveLength(1));
    expect(await writes()[0].clone().json()).toEqual({
      budget: {
        max_budget: 750,
        budget_duration: "1mo",
      },
    });
  });

  it("reopens the saved budget before background reads finish", async () => {
    const updated = {
      ...member,
      updated_at: "2026-11-19T00:00:00Z",
      budget: { ...baseBudget, max_budget: 250 },
    };
    const refresh = Promise.withResolvers<Response>();
    const serve = fetchMock.getMockImplementation()!;
    let saved = false;
    fetchMock.mockImplementation(async (input, init) => {
      const request = input as Request;
      if (!new URL(request.url).pathname.endsWith("/customer_members/member-1")) return serve(input, init);
      if (request.method === "PATCH") {
        saved = true;
        return json(updated);
      }
      return saved ? refresh.promise : json(member);
    });
    const user = await openBudget("member");
    fireEvent.change(await screen.findByRole("spinbutton", { name: "Max Budget (USD)" }), {
      target: { value: "250" },
    });
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: "Edit budget" }));
    try {
      expect(await screen.findByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(250);
      fireEvent.change(screen.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "275" } });
      refresh.resolve(json(updated));
      await waitFor(() => expect(testQueryClient.isFetching()).toBe(0));
      expect(screen.getByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(275);
    } finally {
      refresh.resolve(json(updated));
    }
  });

  it.each([null, "7d"])("preserves the existing reset period %s when saving rate-only limits", async (duration) => {
    const inheritedMember = {
      ...member,
      budget: { budget_duration: duration },
      budget_override: null,
      budget_override_expires_at: null,
    };
    fetchMock.mockImplementation(async (input) =>
      json(
        new URL((input as Request).url).pathname.endsWith("/customer_members/member-1")
          ? inheritedMember
          : memberOrganization,
      ),
    );
    const user = await openBudget("member");
    await chooseSelectOption(
      user,
      await screen.findByRole("combobox", { name: "Member limits" }),
      "Enter custom limits instead",
    );
    fireEvent.change(screen.getByRole("spinbutton", { name: "Tokens per minute" }), { target: { value: "2000" } });
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(writes()).toHaveLength(1));
    expect(await writes()[0].clone().json()).toEqual({
      budget: {
        tpm_limit: 2000,
      },
    });
  });

  it("saves an edited reset period after clearing the custom spend cap", async () => {
    const custom = { ...member, budget: { max_budget: 20, budget_duration: "7d", tpm_limit: 100 } };
    fetchMock.mockImplementation(async (input) =>
      json(new URL((input as Request).url).pathname.endsWith("/customer_members/member-1") ? custom : organization),
    );
    const user = await openBudget("member");
    fireEvent.change(await screen.findByRole("spinbutton", { name: "Max Budget (USD)" }), {
      target: { value: "" },
    });
    fireEvent.change(screen.getByRole("textbox", { name: "Custom period" }), { target: { value: "14d" } });
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(writes()).toHaveLength(1));
    expect(await writes()[0].clone().json()).toEqual({
      budget: { max_budget: null, budget_duration: "14d" },
    });
  });
});

describe.each(["organization", "member"] as const)("%s budget draft", (scope) => {
  it("discards a cancelled draft and retains unsaved values after a failed save", async () => {
    const { dialog, error } = views[scope];
    const serve = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (input, init) =>
      input instanceof Request && input.method === "PATCH" ? json({ detail: error }, 503) : serve(input, init),
    );
    const user = await openBudget(scope);
    fireEvent.change(await screen.findByRole("spinbutton", { name: "Max Budget (USD)" }), {
      target: { value: "999" },
    });
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(writes()).toHaveLength(0);
    await user.click(screen.getByRole("button", { name: "Edit budget" }));
    expect(await screen.findByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(100);
    fireEvent.change(screen.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "200" } });
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(error);
    expect(screen.getByRole("spinbutton", { name: "Max Budget (USD)" })).toHaveValue(200);
    expect(screen.getByRole("dialog", { name: dialog })).toBeInTheDocument();
  });
});
