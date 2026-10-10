import { describe, expect, it } from "vitest";

import {
  getBatchIdFromRequestId,
  getBatchModels,
  getBatchRequestCounts,
  getReasoningTokens,
  isBatchCallType,
} from "./batchLogUtils";

/** Metadata shape the batch cost poller writes on an aretrieve_batch spend row. */
const batchCostMetadata = {
  batch_models: ["gemini-2.5-flash"],
  batch_successful_requests: 2,
  batch_failed_requests: 1,
  usage_object: {
    total_tokens: 270,
    prompt_tokens: 14,
    completion_tokens: 256,
    completion_tokens_details: { text_tokens: 32, reasoning_tokens: 224 },
  },
};

describe("isBatchCallType", () => {
  it("recognizes the poller's aretrieve_batch and the create call types", () => {
    for (const callType of ["aretrieve_batch", "retrieve_batch", "acreate_batch", "create_batch"]) {
      expect(isBatchCallType(callType)).toBe(true);
    }
    expect(isBatchCallType("acompletion")).toBe(false);
  });
});

describe("getBatchRequestCounts", () => {
  it("reads both counts off a batch cost row", () => {
    expect(getBatchRequestCounts(batchCostMetadata)).toEqual({ successful: 2, failed: 1 });
  });

  it("returns undefined for a non-batch row and for null counts, so no rollup renders", () => {
    expect(getBatchRequestCounts({ status: "success" })).toBeUndefined();
    expect(getBatchRequestCounts({ batch_successful_requests: null, batch_failed_requests: null })).toBeUndefined();
    expect(getBatchRequestCounts(undefined)).toBeUndefined();
  });

  it("treats a lone present count as the other being 0, for rows logged mid-rollout", () => {
    expect(getBatchRequestCounts({ batch_successful_requests: 3 })).toEqual({ successful: 3, failed: 0 });
  });
});

describe("getBatchIdFromRequestId", () => {
  it("strips the poller's synthetic _batch_cost suffix down to the provider batch id", () => {
    expect(getBatchIdFromRequestId("batch_abc123_batch_cost")).toBe("batch_abc123");
  });

  it("returns undefined for ordinary request ids and a bare suffix", () => {
    expect(getBatchIdFromRequestId("chatcmpl-123")).toBeUndefined();
    expect(getBatchIdFromRequestId("_batch_cost")).toBeUndefined();
  });
});

describe("getBatchModels", () => {
  it("returns the model list from metadata.batch_models", () => {
    expect(getBatchModels(batchCostMetadata)).toEqual(["gemini-2.5-flash"]);
  });

  it("returns undefined when absent, null, or empty", () => {
    expect(getBatchModels({})).toBeUndefined();
    expect(getBatchModels({ batch_models: null })).toBeUndefined();
    expect(getBatchModels({ batch_models: [] })).toBeUndefined();
  });
});

describe("getReasoningTokens", () => {
  it("reads reasoning tokens from usage_object on a batch cost row", () => {
    expect(getReasoningTokens(batchCostMetadata)).toBe(224);
  });

  it("prefers additional_usage_values, which per-request rows carry", () => {
    const metadata = {
      additional_usage_values: { completion_tokens_details: { reasoning_tokens: 40 } },
      usage_object: { completion_tokens_details: { reasoning_tokens: 999 } },
    };
    expect(getReasoningTokens(metadata)).toBe(40);
  });

  it("returns undefined when the breakout is null or missing", () => {
    expect(getReasoningTokens({ usage_object: { completion_tokens_details: null } })).toBeUndefined();
    expect(getReasoningTokens({})).toBeUndefined();
    expect(getReasoningTokens(undefined)).toBeUndefined();
  });
});
