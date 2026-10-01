import { describe, expect, it } from "vitest";

import { filterModelActivity, modelActivityMatches } from "./modelActivityFilter";
import type { ModelActivityData } from "./types";

function activity(label: string): ModelActivityData {
  return {
    label,
    total_requests: 1,
    total_successful_requests: 1,
    total_failed_requests: 0,
    total_cache_read_input_tokens: 0,
    total_cache_creation_input_tokens: 0,
    total_tokens: 10,
    prompt_tokens: 5,
    completion_tokens: 5,
    total_spend: 0.01,
    top_api_keys: [],
    top_models: [],
    daily_data: [],
  };
}

const gpt = activity("GPT-5.5");
const claude = activity("Claude Sonnet");

const modelMetrics: Record<string, ModelActivityData> = {
  "gpt-5.5": gpt,
  "claude-sonnet-4": claude,
};

describe("modelActivityMatches", () => {
  it("matches every model on an empty or whitespace query", () => {
    expect(modelActivityMatches("gpt-5.5", gpt, "")).toBe(true);
    expect(modelActivityMatches("gpt-5.5", gpt, "   ")).toBe(true);
  });

  it("matches the model key case-insensitively", () => {
    expect(modelActivityMatches("gpt-5.5", gpt, "GPT-5")).toBe(true);
    expect(modelActivityMatches("claude-sonnet-4", claude, "gpt-5")).toBe(false);
  });

  it("matches the model label case-insensitively", () => {
    expect(modelActivityMatches("claude-sonnet-4", claude, "sonnet")).toBe(true);
    expect(modelActivityMatches("gpt-5.5", gpt, "sonnet")).toBe(false);
  });

  it("trims surrounding whitespace from the query", () => {
    expect(modelActivityMatches("gpt-5.5", gpt, "  gpt-5.5  ")).toBe(true);
  });
});

describe("filterModelActivity", () => {
  it("returns the same object when the query is blank", () => {
    expect(filterModelActivity(modelMetrics, "")).toBe(modelMetrics);
  });

  it("keeps only the models matching the query, preserving their keys", () => {
    expect(filterModelActivity(modelMetrics, "sonnet")).toEqual({ "claude-sonnet-4": claude });
    expect(Object.keys(filterModelActivity(modelMetrics, "gpt"))).toEqual(["gpt-5.5"]);
  });

  it("returns an empty record when nothing matches", () => {
    expect(filterModelActivity(modelMetrics, "llama")).toEqual({});
  });
});
