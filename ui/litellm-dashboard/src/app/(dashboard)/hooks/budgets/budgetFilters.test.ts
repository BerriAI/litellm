import type { ColumnFiltersState } from "@tanstack/react-table";
import { describe, expect, it } from "vitest";

import { joinArrayFilters, splitArrayFilters } from "../common/useResourceList";
import {
  BUDGET_DURATION_UNSET,
  BUDGET_LIST_URL_STATE,
  budgetFiltersFromUrl,
  budgetFiltersToUrl,
  serializeBudgetFilters,
} from "./budgetFilters";

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

const urlEntries = (filters: ColumnFiltersState): Record<string, unknown> =>
  Object.fromEntries(budgetFiltersToUrl(filters).map((filter) => [filter.id, filter.value]));

const ARRAY_COLUMNS = BUDGET_LIST_URL_STATE.arrayFilterColumns ?? [];

const throughUrl = (filters: ColumnFiltersState): ColumnFiltersState =>
  budgetFiltersFromUrl(splitArrayFilters(joinArrayFilters(budgetFiltersToUrl(filters), ARRAY_COLUMNS), ARRAY_COLUMNS));

describe("budgetFiltersToUrl", () => {
  it("flattens every drawer value onto its own URL slot", () => {
    const slots: Record<string, unknown> = {
      duration: ["7d", "30d"],
      max_min: "10",
      max_max: "250.5",
      unlimited: "",
      created_from: "2026-01-05",
      created_to: "2026-01-06",
    };
    expect(
      urlEntries([
        { id: "budget_duration", value: ["7d", "30d"] },
        { id: "max_budget", value: { min: "10", max: "250.5" } },
        { id: "created_at", value: { from: "2026-01-05", to: "2026-01-06" } },
      ]),
    ).toEqual(slots);
  });

  it("stores 'Unlimited only' as a flag", () => {
    expect(urlEntries([{ id: "max_budget", value: { unlimitedOnly: true } }])).toMatchObject({
      unlimited: "true",
      max_min: "",
      max_max: "",
    });
  });

  it("blanks every slot when no filter is active, so each URL key gets cleared", () => {
    const slots = budgetFiltersToUrl([]);
    expect(slots.map((slot) => slot.id)).toEqual(BUDGET_LIST_URL_STATE.filterColumns);
    expect(slots.every((slot) => slot.value === "" || (Array.isArray(slot.value) && slot.value.length === 0))).toBe(
      true,
    );
  });
});

describe("budgetFiltersFromUrl", () => {
  it("rebuilds the drawer values from the URL slots", () => {
    expect(
      budgetFiltersFromUrl([
        { id: "duration", value: ["24h"] },
        { id: "max_min", value: "5" },
        { id: "created_to", value: "2026-03-01" },
      ]),
    ).toEqual([
      { id: "budget_duration", value: ["24h"] },
      { id: "max_budget", value: { min: "5" } },
      { id: "created_at", value: { to: "2026-03-01" } },
    ]);
  });

  it("drops durations the drawer does not offer", () => {
    expect(budgetFiltersFromUrl([{ id: "duration", value: ["7d", "fortnightly"] }])).toEqual([
      { id: "budget_duration", value: ["7d"] },
    ]);
    expect(budgetFiltersFromUrl([{ id: "duration", value: ["fortnightly"] }])).toEqual([]);
  });

  it("keeps 'Not set' exclusive, matching what the request sends", () => {
    expect(budgetFiltersFromUrl([{ id: "duration", value: ["7d", BUDGET_DURATION_UNSET] }])).toEqual([
      { id: "budget_duration", value: [BUDGET_DURATION_UNSET] },
    ]);
  });

  it("lets the unlimited flag win over a range, as the drawer does", () => {
    expect(
      budgetFiltersFromUrl([
        { id: "max_min", value: "5" },
        { id: "unlimited", value: "true" },
      ]),
    ).toEqual([{ id: "max_budget", value: { unlimitedOnly: true } }]);
  });

  it("ignores hand-edited amounts and days that are not valid", () => {
    expect(
      budgetFiltersFromUrl([
        { id: "max_min", value: "lots" },
        { id: "max_max", value: "20" },
        { id: "unlimited", value: "yes" },
        { id: "created_from", value: "01/05/2026" },
        { id: "created_to", value: "2026-99-99" },
      ]),
    ).toEqual([{ id: "max_budget", value: { max: "20" } }]);
  });

  it.each(["0x10", "0b1", "0o7", "Infinity", "1_000", "1e", "."])(
    "drops the amount %s, which the route cannot parse as a decimal",
    (amount) => {
      expect(budgetFiltersFromUrl([{ id: "max_min", value: amount }])).toEqual([]);
    },
  );

  it.each(["0", "99.5", ".5", "7.", "1e3", "-2"])("keeps the decimal amount %s", (amount) => {
    expect(budgetFiltersFromUrl([{ id: "max_min", value: amount }])).toEqual([
      { id: "max_budget", value: { min: amount } },
    ]);
  });
});

describe("budget filters through the URL", () => {
  it.each<[string, ColumnFiltersState]>([
    ["durations", [{ id: "budget_duration", value: ["1h", "30d"] }]],
    ["not set", [{ id: "budget_duration", value: [BUDGET_DURATION_UNSET] }]],
    ["a max budget range", [{ id: "max_budget", value: { min: "0", max: "99.5" } }]],
    ["unlimited only", [{ id: "max_budget", value: { unlimitedOnly: true } }]],
    ["a created range", [{ id: "created_at", value: { from: "2026-01-05", to: "2026-01-06" } }]],
    [
      "everything at once",
      [
        { id: "budget_duration", value: ["7d"] },
        { id: "max_budget", value: { max: "10" } },
        { id: "created_at", value: { from: "2026-02-01" } },
      ],
    ],
  ])("restores %s exactly and sends the same request", (_, filters) => {
    const restored = throughUrl(filters);
    expect(restored).toEqual(filters);
    expect(serializeBudgetFilters(restored)).toEqual(serializeBudgetFilters(filters));
  });

  it("stores the reset durations comma separated", () => {
    const stored = joinArrayFilters(
      budgetFiltersToUrl([{ id: "budget_duration", value: ["7d", "30d"] }]),
      ARRAY_COLUMNS,
    );
    expect(stored.find((slot) => slot.id === "duration")?.value).toBe("7d,30d");
  });
});
