import { describe, expect, it } from "vitest";

import { filterModelActivity } from "./modelActivityFilter";
import type { ModelActivityData } from "./types";

const activity = (label: string): ModelActivityData => ({
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
  top_models: [],
  daily_data: [],
});

const modelMetrics: Record<string, ModelActivityData> = {
  "openai/gpt-4o": activity("GPT-4o"),
  "anthropic/claude-3": activity("Claude 3 Sonnet"),
};

describe("filterModelActivity", () => {
  it("matches by model key", () => {
    expect(Object.keys(filterModelActivity(modelMetrics, "openai"))).toEqual(["openai/gpt-4o"]);
  });

  it("matches by model label", () => {
    expect(Object.keys(filterModelActivity(modelMetrics, "Sonnet"))).toEqual(["anthropic/claude-3"]);
  });

  it("matches case-insensitively", () => {
    expect(Object.keys(filterModelActivity(modelMetrics, "GPT-4O"))).toEqual(["openai/gpt-4o"]);
  });

  it("returns all models unchanged for a whitespace-only query", () => {
    expect(filterModelActivity(modelMetrics, "   ")).toBe(modelMetrics);
  });

  it("returns an empty record when nothing matches", () => {
    expect(filterModelActivity(modelMetrics, "unknown")).toEqual({});
  });
});
