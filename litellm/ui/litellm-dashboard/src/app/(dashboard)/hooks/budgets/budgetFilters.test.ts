import { describe, expect, it } from "vitest";

import { BUDGET_DURATION_UNSET, serializeBudgetFilters } from "./budgetFilters";

describe("serializeBudgetFilters", () => {
  it("sends nothing when no filter is active", () => {
    expect(serializeBudgetFilters([])).toEqual({});
  });

  it("maps selected durations onto the in operator", () => {
    expect(serializeBudgetFilters([{ id: "budget_duration", value: ["7d", "30d"] }])).toEqual({
      "filter[budget_duration][in]": "7d,30d",
    });
  });

  it("maps 'Not set' onto is_null instead of in", () => {
    expect(serializeBudgetFilters([{ id: "budget_duration", value: [BUDGET_DURATION_UNSET] }])).toEqual({
      "filter[budget_duration][is_null]": "true",
    });
  });

  it("never sends in alongside is_null for the same field", () => {
    const params = serializeBudgetFilters([{ id: "budget_duration", value: ["7d", BUDGET_DURATION_UNSET] }]);
    expect(params["filter[budget_duration][in]"]).toBeUndefined();
    expect(params["filter[budget_duration][is_null]"]).toBe("true");
  });

  it("maps a max budget range onto gte and lte", () => {
    expect(serializeBudgetFilters([{ id: "max_budget", value: { min: "10", max: "250.5" } }])).toEqual({
      "filter[max_budget][gte]": "10",
      "filter[max_budget][lte]": "250.5",
    });
  });

  it("sends only the bound that was filled in", () => {
    expect(serializeBudgetFilters([{ id: "max_budget", value: { min: "10", max: "" } }])).toEqual({
      "filter[max_budget][gte]": "10",
    });
  });

  it("maps 'Unlimited only' onto is_null and drops the range", () => {
    const params = serializeBudgetFilters([{ id: "max_budget", value: { min: "10", unlimitedOnly: true } }]);
    expect(params).toEqual({ "filter[max_budget][is_null]": "true" });
  });

  it("widens a created-at day range to cover the whole local days", () => {
    const params = serializeBudgetFilters([{ id: "created_at", value: { from: "2026-01-05", to: "2026-01-06" } }]);
    expect(params["filter[created_at][gte]"]).toBe(new Date("2026-01-05T00:00:00.000").toISOString());
    expect(params["filter[created_at][lte]"]).toBe(new Date("2026-01-06T23:59:59.999").toISOString());
  });

  it("ignores an unparseable date rather than sending a broken bound", () => {
    expect(serializeBudgetFilters([{ id: "created_at", value: { from: "not-a-date" } }])).toEqual({});
  });

  it("ignores filter ids the route does not declare", () => {
    expect(serializeBudgetFilters([{ id: "spend", value: "5" }])).toEqual({});
  });
});
