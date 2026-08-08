import { describe, expect, it } from "vitest";
import { formItemValidateJSONObject, hasRoutingStrategyArgs, routingStrategyLabel } from "./routing_strategy_options";

describe("formItemValidateJSONObject", () => {
  it("accepts empty input and JSON objects", async () => {
    await expect(formItemValidateJSONObject(undefined, "")).resolves.toBeUndefined();
    await expect(formItemValidateJSONObject(undefined, '{"ttl": 3600}')).resolves.toBeUndefined();
    await expect(formItemValidateJSONObject(undefined, "{}")).resolves.toBeUndefined();
  });

  it("rejects JSON that is not an object", async () => {
    await expect(formItemValidateJSONObject(undefined, "[1, 2]")).rejects.toMatch(/JSON object/);
    await expect(formItemValidateJSONObject(undefined, "3600")).rejects.toMatch(/JSON object/);
    await expect(formItemValidateJSONObject(undefined, "null")).rejects.toMatch(/JSON object/);
  });

  it("rejects invalid JSON", async () => {
    await expect(formItemValidateJSONObject(undefined, "{ttl:")).rejects.toMatch(/valid JSON/);
  });
});

describe("hasRoutingStrategyArgs", () => {
  it("treats empty and missing objects as unset", () => {
    expect(hasRoutingStrategyArgs(undefined)).toBe(false);
    expect(hasRoutingStrategyArgs(null)).toBe(false);
    expect(hasRoutingStrategyArgs({})).toBe(false);
    expect(hasRoutingStrategyArgs({ ttl: 3600 })).toBe(true);
  });
});

describe("routingStrategyLabel", () => {
  it("labels known strategies and defaults the rest", () => {
    expect(routingStrategyLabel("cost-based-routing")).toContain("Cost-Based");
    expect(routingStrategyLabel(undefined)).toBe("Inherit router default");
    expect(routingStrategyLabel("")).toBe("Inherit router default");
  });
});
