import { describe, expect, it } from "vitest";

import { endUserBudgetIdUpdate, keyOffersEndUserBudget, storedEndUserBudgetId } from "./endUserBudgetPayload";

describe("keyOffersEndUserBudget", () => {
  it("offers the control on service account keys and on keys that already carry a budget", () => {
    expect(keyOffersEndUserBudget({ service_account_id: "svc-a" })).toBe(true);
    expect(keyOffersEndUserBudget({ end_user_budget_id: "svc-a-budget" })).toBe(true);
  });

  it.each([undefined, null, {}, { service_account_id: "" }, { tags: ["x"] }])("hides it for %j", (metadata) => {
    expect(keyOffersEndUserBudget(metadata)).toBe(false);
  });
});

describe("storedEndUserBudgetId", () => {
  it("reads the budget id a key applies to the customers it creates", () => {
    expect(storedEndUserBudgetId({ service_account_id: "svc-a", end_user_budget_id: "svc-a-budget" })).toBe(
      "svc-a-budget",
    );
  });

  it.each([undefined, null, "not-an-object", [], {}, { end_user_budget_id: 7 }])(
    "reads %j as no default budget",
    (metadata) => {
      expect(storedEndUserBudgetId(metadata)).toBe("");
    },
  );
});

describe("endUserBudgetIdUpdate", () => {
  it("leaves the field off the payload when the selection matches the stored value", () => {
    expect(endUserBudgetIdUpdate("svc-a-budget", "svc-a-budget")).toBeUndefined();
    expect(endUserBudgetIdUpdate(null, "")).toBeUndefined();
  });

  it("sends the newly selected budget id", () => {
    expect(endUserBudgetIdUpdate("svc-b-budget", "svc-a-budget")).toBe("svc-b-budget");
    expect(endUserBudgetIdUpdate("svc-a-budget", "")).toBe("svc-a-budget");
  });

  it("sends an empty string so the backend clears a previously stored budget", () => {
    expect(endUserBudgetIdUpdate(null, "svc-a-budget")).toBe("");
  });
});
