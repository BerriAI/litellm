import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { buildRequest, compactNumber, metricValue, recommenderParsers, requestRecommendations } from "./api";

beforeEach(() => registerBaseUrlGetter(() => "http://localhost:4000"));
afterEach(() => {
  vi.unstubAllGlobals();
  registerBaseUrlGetter(() => "");
});

describe("recommender request state", () => {
  it("sends priorities, focus and hard requirements without dropping zero values", () => {
    const state: Parameters<typeof buildRequest>[0] = {
      intelligence: 0,
      speed: 3,
      cost: 1,
      focus: ["coding", "agentic"],
      reasoning: true,
      vision: false,
      function_calling: true,
      context: 128000,
    };
    const expected = {
      priorities: { intelligence: 0, speed: 3, cost: 1 },
      focus: ["coding", "agentic"],
      requirements: { reasoning: true, vision: false, function_calling: true, min_context_tokens: 128000 },
      limit: 3,
    };
    expect(buildRequest(state)).toEqual(expected);
  });
  it("rejects malformed query values rather than sending them to the gateway", () => {
    expect(recommenderParsers.cost.parse("4")).toBeNull();
    expect(recommenderParsers.cost.parse("0")).toBe(0);
    expect(recommenderParsers.context.parse("-1")).toBeNull();
    expect(recommenderParsers.context.parse("128000")).toBe(128000);
  });
  it("labels unknown signals and formats raw metrics", () => {
    expect(metricValue("intelligence", null)).toBe("Unknown");
    expect(metricValue("intelligence", 0.71)).toBe("71% benchmark score");
    expect(metricValue("cost", 0.2)).toBe("$0.20 / 1M");
    expect(metricValue("speed", 1.24)).toBe("1.2s");
    expect(compactNumber(922000)).toBe("922K");
    expect(compactNumber(1000000)).toBe("1M");
  });
  it("returns an empty shortlist without inventing results", async () => {
    const response = { recommendations: [], candidates_considered: 2, candidates_filtered: 2, data_sources: [] };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(response), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    expect(await requestRecommendations({}, "test-key")).toEqual(response);
  });
  it("surfaces gateway errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Unavailable" }), {
          status: 503,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    await expect(requestRecommendations({}, "test-key")).rejects.toThrow("Unavailable");
  });
});
