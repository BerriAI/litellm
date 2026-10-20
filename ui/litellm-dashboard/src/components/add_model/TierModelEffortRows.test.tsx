import { describe, expect, it } from "vitest";
import { tierEffortRows } from "./TierModelEffortRows";

describe("tierEffortRows", () => {
  it("offers the levels the proxy reports for the model", () => {
    const rows = tierEffortRows({
      models: ["gpt-5-mini"],
      effortOptionsByModel: { "gpt-5-mini": ["low", "medium", "high"] },
      paramsByModel: undefined,
    });

    expect(rows).toEqual([{ model: "gpt-5-mini", effort: undefined, options: ["low", "medium", "high"] }]);
  });

  it("keeps a row whose model reports no level but already has an effort stored, so it can be cleared", () => {
    const rows = tierEffortRows({
      models: ["off-map-model"],
      effortOptionsByModel: {},
      paramsByModel: { "off-map-model": { reasoning_effort: "high" } },
    });

    expect(rows).toEqual([{ model: "off-map-model", effort: "high", options: ["high"] }]);
  });

  it("drops a row whose model reports no level and has nothing stored", () => {
    const rows = tierEffortRows({
      models: ["plain-chat-model"],
      effortOptionsByModel: { "plain-chat-model": [] },
      paramsByModel: { "plain-chat-model": { temperature: 0.5 } },
    });

    expect(rows).toEqual([]);
  });

  it("lists a stored level the model no longer reports without duplicating the ones it does", () => {
    const rows = tierEffortRows({
      models: ["gpt-5-mini"],
      effortOptionsByModel: { "gpt-5-mini": ["low", "high"] },
      paramsByModel: { "gpt-5-mini": { reasoning_effort: "xhigh" } },
    });

    expect(rows).toEqual([{ model: "gpt-5-mini", effort: "xhigh", options: ["low", "high", "xhigh"] }]);
  });

  it("does not repeat a stored level the model already reports", () => {
    const rows = tierEffortRows({
      models: ["gpt-5-mini"],
      effortOptionsByModel: { "gpt-5-mini": ["low", "high"] },
      paramsByModel: { "gpt-5-mini": { reasoning_effort: "high" } },
    });

    expect(rows).toEqual([{ model: "gpt-5-mini", effort: "high", options: ["low", "high"] }]);
  });

  it.each([
    ["an unset key", {}],
    ["an explicit null", { reasoning_effort: null }],
    ["an empty string", { reasoning_effort: "" }],
  ])("reads %s as no stored effort", (_label, params) => {
    const rows = tierEffortRows({
      models: ["gpt-5-mini"],
      effortOptionsByModel: { "gpt-5-mini": ["low"] },
      paramsByModel: { "gpt-5-mini": params },
    });

    expect(rows[0].effort).toBeUndefined();
  });

  it("renders a non-string stored value as a string so the select can show and clear it", () => {
    const rows = tierEffortRows({
      models: ["gpt-5-mini"],
      effortOptionsByModel: { "gpt-5-mini": ["low"] },
      paramsByModel: { "gpt-5-mini": { reasoning_effort: 3 } },
    });

    expect(rows).toEqual([{ model: "gpt-5-mini", effort: "3", options: ["low", "3"] }]);
  });
});
