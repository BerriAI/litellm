import { describe, expect, it } from "vitest";
import { toRetryPolicyPayload, toRetrySettings } from "./mapper";

describe("retry settings payload", () => {
  it("uses null for missing policy maps without sending the inherited retry count", () => {
    const { draft, defaultRetry } = toRetrySettings({ num_retries: 0 });

    expect(defaultRetry).toBe(0);
    expect(toRetryPolicyPayload(draft)).toEqual({ retry_policy: null, model_group_retry_policy: null });
  });

  it("keeps empty maps, explicit zero overrides and model names verbatim", () => {
    const modelPolicies = { ["__proto__"]: { RateLimitErrorRetries: 0 }, "team.model": {} };
    const { draft } = toRetrySettings({ retry_policy: {}, model_group_retry_policy: modelPolicies });

    expect(JSON.stringify(toRetryPolicyPayload(draft))).toBe(
      JSON.stringify({ retry_policy: {}, model_group_retry_policy: modelPolicies }),
    );
  });
});
