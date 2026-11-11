import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { chooseSelectOption } from "../../../../../tests/test-utils";
import AssignBudgetModal from "./AssignBudgetModal";

const budget: budgetItem = {
  budget_id: "qa-budget",
  max_budget: 50,
  budget_duration: "30d",
  soft_budget: null,
  tpm_limit: null,
  rpm_limit: null,
  tpd_limit: null,
  budget_reset_at: null,
  created_at: null,
  updated_at: null,
};
const fetchMock = vi.fn<typeof fetch>();

function renderDialog() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const onClose = vi.fn();
  render(
    <QueryClientProvider client={client}>
      <AssignBudgetModal budget={budget} accessToken="qa-token" onClose={onClose} />
    </QueryClientProvider>,
  );
  return { onClose };
}

function writes() {
  return fetchMock.mock.calls
    .filter(([, options]) => options?.method === "POST")
    .map(([url, options]) => ({
      path: new URL(String(url), "http://localhost").pathname,
      body: JSON.parse(String(options?.body)) as unknown,
    }));
}

describe("budget assignment form", () => {
  beforeEach(() => {
    const token = ["e30", btoa(JSON.stringify({ key: "qa-token" })), "fixture"].join(".");
    document.cookie = `token=${token}; path=/`;
    fetchMock.mockReset();
    fetchMock.mockImplementation(async (input) => {
      const url = new URL(input instanceof Request ? input.url : String(input), "http://localhost");
      const path = url.pathname;
      const responses: Record<string, object> = {
        "/management/v1/users/me": { user_id: "qa-admin", user_role: "proxy_admin" },
        "/litellm/.well-known/litellm-ui-config": { admin_ui_disabled: false },
        "/customer/list":
          url.searchParams.get("search") === "Support"
            ? [{ user_id: "qa-end-user", alias: "Support", spend: 0, blocked: false }]
            : [{ user_id: "qa-other", alias: "First page", spend: 0, blocked: false }],
        "/team/list": [{ team_id: "qa-team", team_alias: "Engineering" }],
        "/config/list": [{ field_name: "request_pattern_tag_mappings", field_value: [], editable: true }],
        "/tag/list": [],
      };
      return Response.json(responses[path] ?? {});
    });
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    document.cookie = "token=; max-age=0; path=/";
    vi.unstubAllGlobals();
  });

  it("finds an existing customer by name outside the initial page", async () => {
    const user = userEvent.setup();
    const { onClose } = renderDialog();
    await screen.findByText("First page", { selector: "option" });
    fireEvent.change(screen.getByLabelText("Customer name or end-user ID"), { target: { value: "Support" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "Assign budget" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Assign budget" }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(writes()).toEqual([
      { path: "/customer/update", body: { user_id: "qa-end-user", budget_id: budget.budget_id } },
    ]);
  });

  it("saves the chosen user-agent pattern rather than treating it as an end user", async () => {
    const user = userEvent.setup();
    const { onClose } = renderDialog();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Assign to" }), "User-agent pattern");
    fireEvent.change(screen.getByLabelText("Pattern"), { target: { value: "claude-code/*" } });
    await user.click(screen.getByRole("button", { name: "Assign budget" }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(writes()).toEqual([
      { path: "/tag/new", body: { name: "budget:qa-budget", budget_id: budget.budget_id } },
      {
        path: "/config/field/update",
        body: {
          config_type: "general_settings",
          field_name: "request_pattern_tag_mappings",
          field_value: [{ source: "header", field: "user-agent", pattern: "claude-code/*", tag: "budget:qa-budget" }],
        },
      },
    ]);
  });

  it("copies the selected budget to the chosen team", async () => {
    const user = userEvent.setup();
    const { onClose } = renderDialog();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Assign to" }), "Team / group");
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Team" }), "Engineering");
    await user.click(screen.getByRole("button", { name: "Assign budget" }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(writes()).toEqual([
      {
        path: "/team/update",
        body: {
          team_id: "qa-team",
          max_budget: budget.max_budget,
          budget_duration: budget.budget_duration,
          soft_budget: null,
          tpm_limit: null,
          rpm_limit: null,
          tpd_limit: null,
        },
      },
    ]);
  });
});
