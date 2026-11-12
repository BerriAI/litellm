import { describe, expect, it } from "vitest";
import type { components } from "@/lib/http/schema";
import { deprecationUsageRange, usedDeprecationAlerts } from "./deprecationAlerts";

type Model = components["schemas"]["ModelDeprecationInfo"];
type Day = components["schemas"]["DailySpendData"];
const model: Model = {
  model_name: "support",
  litellm_model: "anthropic/claude-sonnet-4-5",
  deprecation_date: "2026-11-30",
  days_until_deprecation: 26,
  status: "imminent",
};
// Use the API's defaults for the unrelated aggregate metrics.
const metrics: components["schemas"]["SpendMetrics"] = {
  spend: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
};
function day(date: string, count: number, name = model.litellm_model!): Day {
  return { date, metrics, breakdown: { models: { [name]: { metrics: { ...metrics, api_requests: count } } } } };
}

describe("usedDeprecationAlerts", () => {
  it("should aggregate requests and report the latest day of actual model usage", () => {
    const result = usedDeprecationAlerts([model], [day("2026-11-04", 12), day("2026-10-20", 8)]);
    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({ usageModel: model.litellm_model, requests: 20, lastUsedDate: "2026-11-04" });
  });
  it("should omit configured models without observed requests", () => {
    expect(usedDeprecationAlerts([model], [])).toEqual([]);
    expect(usedDeprecationAlerts([model], [day("2026-11-04", 0)])).toEqual([]);
  });
  it("should avoid attributing healthy deployments in the same group to a deprecated model", () => {
    const activity = day("2026-11-04", 100, "anthropic/claude-sonnet-4-6");
    activity.breakdown!.model_groups = { support: { metrics: { ...metrics, api_requests: 100 } } };
    expect(usedDeprecationAlerts([model], [activity])).toEqual([]);
  });
  it("should match provider-qualified usage when the cost map names an unqualified model", () => {
    const entry = { ...model, litellm_model: "claude-sonnet-4-5", litellm_provider: "anthropic" };
    const result = usedDeprecationAlerts([entry], [day("2026-11-04", 12)]);
    expect(result[0]).toMatchObject({ usageModel: "anthropic/claude-sonnet-4-5", requests: 12 });
  });
  it("should keep usage for different providers separate", () => {
    const entry = { ...model, litellm_provider: "anthropic" };
    expect(usedDeprecationAlerts([entry], [day("2026-11-04", 12, "bedrock/claude-sonnet-4-5")])).toEqual([]);
  });
  it("should deduplicate deployments and put the earliest retirement first", () => {
    const retired = { ...model, deprecation_date: "2026-10-23", status: "deprecated" as const };
    const result = usedDeprecationAlerts([model, retired, model], [day("2026-11-04", 10)]);
    expect(result.map((alert) => alert.model.deprecation_date)).toEqual(["2026-10-23", "2026-11-30"]);
  });
  it("should give an escalated alert a new dismissal identity", () => {
    const activity = [day("2026-11-04", 10)];
    const imminent = usedDeprecationAlerts([model], activity)[0];
    const retired = usedDeprecationAlerts([{ ...model, status: "deprecated" }], activity)[0];
    expect(retired.id).not.toBe(imminent.id);
  });
  it("should use the public model name when underlying model metadata is unavailable", () => {
    const result = usedDeprecationAlerts([{ ...model, litellm_model: null }], [day("2026-11-04", 10, "support")]);
    expect(result[0].requests).toBe(10);
  });
});

describe("deprecationUsageRange", () => {
  it("should include exactly 30 UTC calendar dates across month boundaries", () => {
    expect(deprecationUsageRange(new Date("2026-11-04T23:00:00Z"))).toEqual({
      start_date: "2026-10-06",
      end_date: "2026-11-04",
    });
  });
});

it("retains daily totals, explicit zeroes and gaps for the alert chart", () => {
  const alerts = usedDeprecationAlerts(
    [model],
    [day("2026-11-04", 12), day("2026-11-04", 3), day("2026-11-02", 0), day("2026-11-01", 8, "example-other")],
  );
  expect(alerts[0].history).toEqual([
    { date: "2026-11-02", value: 0 },
    { date: "2026-11-04", value: 15 },
  ]);
});
