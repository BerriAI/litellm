import { describe, expect, it } from "vitest";
import {
  buildBulkMemberPatches,
  bulkMemberLimitsSchema,
  EMPTY_BULK_MEMBER_LIMITS,
  type BulkMemberLimitsFormValues,
} from "./bulkMemberLimits";

const withFields = (fields: Partial<BulkMemberLimitsFormValues>): BulkMemberLimitsFormValues => ({
  ...EMPTY_BULK_MEMBER_LIMITS,
  ...fields,
});

describe("bulkMemberLimitsSchema", () => {
  it("leaves every unticked limit out of the patch, even when its input holds a value", () => {
    const parsed = bulkMemberLimitsSchema.parse(
      withFields({
        max_budget_in_team: { change: false, value: "not a number" },
        allowed_models: { change: false, value: ["gpt-5"] },
      }),
    );

    expect(parsed).toEqual({});
  });

  it("turns a ticked blank into null so the limit is cleared", () => {
    const allTickedBlank: BulkMemberLimitsFormValues = {
      max_budget_in_team: { change: true, value: "  " },
      budget_duration: { change: true, value: "" },
      tpm_limit: { change: true, value: "" },
      rpm_limit: { change: true, value: "" },
      allowed_models: { change: true, value: [] },
    };
    const allCleared = {
      max_budget_in_team: null,
      budget_duration: null,
      tpm_limit: null,
      rpm_limit: null,
      allowed_models: null,
    };

    expect(bulkMemberLimitsSchema.parse(allTickedBlank)).toEqual(allCleared);
  });

  it("parses ticked values into the types the endpoint expects, keeping 0 as 0", () => {
    const allTicked: BulkMemberLimitsFormValues = {
      max_budget_in_team: { change: true, value: "12.5" },
      budget_duration: { change: true, value: "30d" },
      tpm_limit: { change: true, value: "0" },
      rpm_limit: { change: true, value: "60" },
      allowed_models: { change: true, value: ["gpt-5", "claude-opus-5-5"] },
    };
    const patch = {
      max_budget_in_team: 12.5,
      budget_duration: "30d",
      tpm_limit: 0,
      rpm_limit: 60,
      allowed_models: ["gpt-5", "claude-opus-5-5"],
    };

    expect(bulkMemberLimitsSchema.parse(allTicked)).toEqual(patch);
  });

  it.each([
    ["a negative budget", { max_budget_in_team: { change: true, value: "-1" } }],
    ["a non-numeric budget", { max_budget_in_team: { change: true, value: "12e" } }],
    ["a fractional TPM limit", { tpm_limit: { change: true, value: "1.5" } }],
    ["a negative RPM limit", { rpm_limit: { change: true, value: "-3" } }],
    ["an RPM limit too large to send exactly", { rpm_limit: { change: true, value: "9007199254740993" } }],
    ["a TPM limit that overflows to Infinity", { tpm_limit: { change: true, value: "9".repeat(400) } }],
  ] as const)("rejects %s", (_, fields) => {
    expect(bulkMemberLimitsSchema.safeParse(withFields(fields)).success).toBe(false);
  });
});

describe("buildBulkMemberPatches", () => {
  it("addresses each member by user_id, falling back to user_email, and applies the same limits to all", () => {
    const patches = buildBulkMemberPatches(
      [
        { user_id: "u-1", user_email: "one@example.com", role: "user" },
        { user_id: null, user_email: "two@example.com", role: "user" },
      ],
      { rpm_limit: 10, allowed_models: null },
    );

    expect(patches).toEqual([
      { user_id: "u-1", rpm_limit: 10, allowed_models: null },
      { user_email: "two@example.com", rpm_limit: 10, allowed_models: null },
    ]);
  });
});
