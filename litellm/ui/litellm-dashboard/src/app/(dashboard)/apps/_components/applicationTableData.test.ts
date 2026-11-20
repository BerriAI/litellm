import { describe, expect, it } from "vitest";
import { EMPTY_CUSTOMER_USAGE } from "../../customers/_components/customerUsageTotals";
import type { App, Budget } from "./appForms";
import type { ApplicationRow } from "./applications";
import { applicationTableRows, sortApplications } from "./applicationTableData";

const application = (alias: string, budget: Budget | null, spend: number | null): ApplicationRow => {
  const resource = { alias, resource_id: `resource-${alias}`, resource_type: "team" as const, app_id: alias };
  const app: App = {
    ...resource,
    auto_create_customers: true,
    spend: 9999,
    budget,
    budget_id: alias,
    budget_usage:
      budget?.max_budget == null
        ? []
        : [{ source: "scalar", max_budget: budget.max_budget, budget_duration: "1mo", spend, reset_at: null }],
    created_at: "",
    updated_at: "",
    created_by: "",
    updated_by: "",
  };
  return { resource, app };
};
const source = [
  application("Unknown", { max_budget: 30, budget_duration: "1mo" }, null),
  application("Unlimited", null, null),
  application("Beta", { max_budget: 2, budget_duration: "1mo" }, 9),
  application("Alpha", { max_budget: 1000, budget_duration: "1mo" }, 100),
];
const groups = [
  { budget_id: "Alpha", name: "Scale" },
  { budget_id: "Beta", name: "Enterprise" },
];
const usage = new Map([
  [null, { ...EMPTY_CUSTOMER_USAGE, spend: 200 }],
  ["Alpha", { ...EMPTY_CUSTOMER_USAGE, spend: 2, tokens: 1000 }],
  ["Beta", { ...EMPTY_CUSTOMER_USAGE, spend: 100, tokens: 9 }],
]);
const rows = applicationTableRows(source, groups, usage, "Custom");
const names = (items: typeof rows) => items.map(({ resource }) => resource.alias);

describe("application sorting", () => {
  it.each([
    ["application", ["Alpha", "Beta", "Unknown", "Unlimited"], ["Unlimited", "Unknown", "Beta", "Alpha"]],
    ["group", ["Unknown", "Beta", "Alpha", "Unlimited"], ["Alpha", "Beta", "Unknown", "Unlimited"]],
    ["spend", ["Beta", "Alpha", "Unknown", "Unlimited"], ["Alpha", "Beta", "Unknown", "Unlimited"]],
    ["budget", ["Beta", "Unknown", "Alpha", "Unlimited"], ["Alpha", "Unknown", "Beta", "Unlimited"]],
    ["share", ["Alpha", "Beta", "Unknown", "Unlimited"], ["Beta", "Alpha", "Unknown", "Unlimited"]],
    ["tokens", ["Beta", "Alpha", "Unknown", "Unlimited"], ["Alpha", "Beta", "Unknown", "Unlimited"]],
    ["status", ["Alpha", "Beta", "Unknown", "Unlimited"], ["Beta", "Alpha", "Unknown", "Unlimited"]],
  ])("sorts %s in both directions with unavailable values last", (id, ascending, descending) => {
    expect(names(sortApplications(rows, "", [{ id, desc: false }]))).toEqual(ascending);
    expect(names(sortApplications(rows, "", [{ id, desc: true }]))).toEqual(descending);
  });

  it("uses monthly usage without a monthly cap and sorts tied values by name", () => {
    const uncapped = source.map(({ resource, app }) => ({
      resource,
      app: app && { ...app, budget: null, budget_usage: [] },
    }));
    const totals = new Map([...usage, ["Beta", { ...EMPTY_CUSTOMER_USAGE, spend: 2, tokens: 0 }]]);
    const prepared = applicationTableRows(uncapped, groups, totals, "Custom");
    expect(names(sortApplications(prepared, "", [{ id: "spend", desc: true }]))).toEqual([
      "Alpha",
      "Beta",
      "Unknown",
      "Unlimited",
    ]);
    expect(names(sortApplications(prepared, "", [{ id: "tokens", desc: false }]))[0]).toBe("Beta");
    expect(names(sortApplications(rows, "  RESOURCE-BETA  ", []))).toEqual(["Beta"]);
  });

  it("keeps a missing monthly counter unknown and uses the displayed monthly window limit", () => {
    const base = application(
      "Window",
      { max_budget: 1000, budget_duration: "12mo", budget_limits: [{ max_budget: 20, budget_duration: "1mo" }] },
      null,
    );
    const row = { ...base, app: base.app && { ...base.app, budget_usage: null } };
    const totals = new Map([["Window", { ...EMPTY_CUSTOMER_USAGE, spend: 75 }]]);
    expect(applicationTableRows([row], [], totals, "Custom")[0].values).toMatchObject({ spend: null, budget: 20 });
  });
});
