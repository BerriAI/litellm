import { describe, expect, it } from "vitest";
import { buildUsageSearchParams, parseUsageView, USAGE_KEY_PARAM, usageHref } from "./useUsagePageSearchParams";

describe("parseUsageView", () => {
  it("returns global for missing or invalid values", () => {
    expect(parseUsageView(null)).toBe("global");
    expect(parseUsageView(undefined)).toBe("global");
    expect(parseUsageView("")).toBe("global");
    expect(parseUsageView("not-a-view")).toBe("global");
  });

  it("accepts known UsageOption values", () => {
    expect(parseUsageView("my-budgets")).toBe("my-budgets");
    expect(parseUsageView("team")).toBe("team");
    expect(parseUsageView("global")).toBe("global");
  });
});

describe("USAGE_KEY_PARAM", () => {
  it("uses key as the Usage Top Keys query param name", () => {
    expect(USAGE_KEY_PARAM).toBe("key");
  });
});

describe("buildUsageSearchParams", () => {
  it("sets key while preserving existing view", () => {
    const next = buildUsageSearchParams("?view=team", { key: "hashed-key-abc" });
    expect(next.get("view")).toBe("team");
    expect(next.get("key")).toBe("hashed-key-abc");
  });

  it("clears key when null", () => {
    const next = buildUsageSearchParams("view=team&key=abc", { key: null });
    expect(next.get("view")).toBe("team");
    expect(next.has("key")).toBe(false);
  });

  it("keeps view=global in the query string so leaving my-budgets cannot snap back", () => {
    const next = buildUsageSearchParams("view=my-budgets&team=t1", { view: "global", team: null });
    expect(next.get("view")).toBe("global");
    expect(next.has("team")).toBe(false);
  });

  it("omits view only when explicitly cleared with null", () => {
    const next = buildUsageSearchParams("view=team", { view: null });
    expect(next.has("view")).toBe(false);
  });
});

describe("usageHref", () => {
  it("keeps the /ui pathname when adding query params", () => {
    const params = buildUsageSearchParams("view=team", { key: "hashed-key-abc" });
    expect(usageHref("/ui/usage", params)).toBe("/ui/usage?view=team&key=hashed-key-abc");
  });
});
