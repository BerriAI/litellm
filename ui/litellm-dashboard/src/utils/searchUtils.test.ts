import { describe, expect, it } from "vitest";

import { filterBySearchTerm, matchesSearchTerm, rankBySearchRelevance } from "./searchUtils";

describe("matchesSearchTerm", () => {
  it("matches everything on an empty or whitespace-only term", () => {
    expect(matchesSearchTerm("", ["anything"])).toBe(true);
    expect(matchesSearchTerm("   ", ["anything"])).toBe(true);
  });

  it("matches a substring of any field, case-insensitively", () => {
    expect(matchesSearchTerm("BILL", ["Billing Router", "routes invoices"])).toBe(true);
    expect(matchesSearchTerm("invoice", ["Billing Router", "routes invoices"])).toBe(true);
  });

  it("matches when every word appears in some field", () => {
    expect(matchesSearchTerm("router invoices", ["Billing Router", "routes invoices"])).toBe(true);
    expect(matchesSearchTerm("router refunds", ["Billing Router", "routes invoices"])).toBe(false);
  });

  it("returns false when nothing matches", () => {
    expect(matchesSearchTerm("zzzz", ["Billing Router", "routes invoices"])).toBe(false);
  });

  it("ignores null and undefined fields", () => {
    expect(matchesSearchTerm("billing", [null, undefined, "Billing Router"])).toBe(true);
    expect(matchesSearchTerm("billing", [null, undefined])).toBe(false);
  });
});

describe("filterBySearchTerm", () => {
  const agents = [
    { name: "Billing Router", description: "routes invoices" },
    { name: "Support Bot", description: "handles tickets" },
  ];

  it("keeps only items whose fields match", () => {
    expect(filterBySearchTerm(agents, "tickets", (a) => [a.name, a.description])).toEqual([agents[1]]);
  });

  it("returns an empty list when nothing matches", () => {
    expect(filterBySearchTerm(agents, "zzzz", (a) => [a.name, a.description])).toEqual([]);
  });

  it("returns all items for an empty term", () => {
    expect(filterBySearchTerm(agents, "", (a) => [a.name, a.description])).toEqual(agents);
  });
});

describe("rankBySearchRelevance", () => {
  it("orders exact match, then prefix match, then shorter names", () => {
    const items = [{ name: "gpt-4o-mini-transcribe" }, { name: "gpt-4o" }, { name: "chatgpt-4o-latest" }];
    expect(rankBySearchRelevance(items, "gpt-4o", (m) => m.name).map((m) => m.name)).toEqual([
      "gpt-4o",
      "gpt-4o-mini-transcribe",
      "chatgpt-4o-latest",
    ]);
  });

  it("keeps the original order for an empty term", () => {
    const items = [{ name: "b" }, { name: "a" }];
    expect(rankBySearchRelevance(items, "", (m) => m.name)).toEqual(items);
  });
});
