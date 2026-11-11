import { describe, expect, it, vi } from "vitest";

import type { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { createApiClient } from "@/lib/http/client";

import { assignBudget, loadPatternMappings, type BudgetAssignment, type PatternMapping } from "./budgetAssignments";

const budget: budgetItem = {
  budget_id: "qa-budget",
  max_budget: 50,
  budget_duration: "30d",
  soft_budget: 40,
  tpm_limit: 500,
  rpm_limit: 5,
  tpd_limit: 9000,
  budget_reset_at: null,
  created_at: null,
  updated_at: null,
};

const agentAssignment: BudgetAssignment = {
  kind: "pattern",
  source: "header",
  field: "user-agent",
  pattern: "claude-code/*",
};

function server(rules: PatternMapping[] = [], editable = true, tags: object[] = []) {
  const fetchMock = vi.fn<typeof fetch>().mockImplementation(async (input) => {
    const path = new URL(String(input)).pathname;
    const data: Record<string, object> = {
      "/config/list": [{ field_name: "request_pattern_tag_mappings", field_value: rules, editable }],
      "/tag/list": tags,
    };
    return Response.json(data[path] ?? {});
  });
  const client = createApiClient({ getBaseUrl: () => "http://proxy.test", fetchImpl: fetchMock });
  const writes = () =>
    fetchMock.mock.calls
      .filter(([, init]) => init?.method === "POST")
      .map(([input, init]) => ({
        path: new URL(String(input)).pathname,
        body: JSON.parse(String(init?.body)) as unknown,
      }));
  return { client, fetchMock, writes };
}

describe("budget assignment", () => {
  it.each([true, false])(
    "assigns the budget ID without changing other customer fields, existing=%s",
    async (existing) => {
      const api = server();
      await assignBudget(api.client, "qa-token", budget, { kind: "customer", userId: "qa-end-user", existing });
      expect(api.writes()).toEqual([
        {
          path: existing ? "/customer/update" : "/customer/new",
          body: { user_id: "qa-end-user", budget_id: budget.budget_id },
        },
      ]);
    },
  );

  it("copies only budget limits to a team, including an explicit never-reset period", async () => {
    const api = server();
    await assignBudget(
      api.client,
      "qa-token",
      { ...budget, budget_duration: null },
      { kind: "team", teamId: "qa-team" },
    );
    expect(api.writes()).toEqual([
      {
        path: "/team/update",
        body: {
          team_id: "qa-team",
          max_budget: budget.max_budget,
          budget_duration: null,
          soft_budget: budget.soft_budget,
          tpm_limit: budget.tpm_limit,
          rpm_limit: budget.rpm_limit,
          tpd_limit: budget.tpd_limit,
        },
      },
    ]);
  });

  it("creates a budgeted tag and preserves other rules when assigning a user-agent pattern", async () => {
    const other: PatternMapping = { source: "body", field: "user", pattern: "qa-*", tag: "other" };
    const api = server([other]);
    await assignBudget(api.client, "qa-token", budget, agentAssignment);
    expect(api.writes()).toEqual([
      { path: "/tag/new", body: { name: "budget:qa-budget", budget_id: budget.budget_id } },
      {
        path: "/config/field/update",
        body: {
          config_type: "general_settings",
          field_name: "request_pattern_tag_mappings",
          field_value: [
            other,
            { source: "header", field: "user-agent", pattern: "claude-code/*", tag: "budget:qa-budget" },
          ],
        },
      },
    ]);
  });

  it("adds another pattern to the same tag without resetting its spend", async () => {
    const api = server([], true, [{ name: "budget:qa-budget", litellm_budget_table: { budget_id: budget.budget_id } }]);
    const assignment: BudgetAssignment = {
      kind: "pattern",
      source: "body",
      field: "metadata.group",
      pattern: "engineering-*",
    };
    await assignBudget(api.client, "qa-token", budget, assignment);
    expect(api.writes()).toEqual([
      {
        path: "/config/field/update",
        body: {
          config_type: "general_settings",
          field_name: "request_pattern_tag_mappings",
          field_value: [{ source: "body", field: "metadata.group", pattern: "engineering-*", tag: "budget:qa-budget" }],
        },
      },
    ]);
  });

  it("does not overwrite file-managed rules or create an unused tag", async () => {
    const api = server([], false);
    await expect(assignBudget(api.client, "qa-token", budget, agentAssignment)).rejects.toThrow("configuration file");
    expect(api.writes()).toEqual([]);
  });

  it("refuses to overwrite malformed rules", async () => {
    const api = server();
    api.fetchMock.mockResolvedValueOnce(
      Response.json([{ field_name: "request_pattern_tag_mappings", field_value: [{ source: "invalid" }] }]),
    );
    await expect(loadPatternMappings(api.client, "qa-token")).rejects.toThrow();
    expect(api.writes()).toEqual([]);
  });

  it("does not claim success if saving the rule fails after creating its tag", async () => {
    const api = server();
    api.fetchMock
      .mockResolvedValueOnce(Response.json([{ field_name: "request_pattern_tag_mappings", field_value: [] }]))
      .mockResolvedValueOnce(Response.json([]))
      .mockResolvedValueOnce(Response.json({}))
      .mockResolvedValueOnce(Response.json({ detail: "Storage unavailable" }, { status: 503 }));
    await expect(assignBudget(api.client, "qa-token", budget, agentAssignment)).rejects.toThrow("Storage unavailable");
  });
});
