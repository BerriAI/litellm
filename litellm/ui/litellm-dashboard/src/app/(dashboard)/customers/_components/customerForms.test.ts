import { describe, expect, test } from "vitest";
import {
  buildCreateCustomerBody,
  buildCustomerBudgetBody,
  buildCustomerFallbacks,
  buildUpdateCustomerBody,
  customerFallbackFields,
  hasIncompleteCustomerFallbacks,
  modelRateLimitFields,
  parseRateLimit,
  formatRateLimit,
  hasInvalidCustomerRateLimits,
} from "./customerForms";

describe("buildCreateCustomerBody", () => {
  const fields = { userId: " acme ", alias: " Acme Corp ", budget: "25", tpm: "1000", rpm: "60" };
  const emptyFields = { userId: "acme", alias: "", budget: "", tpm: "", rpm: "" };

  test("should trim ids and aliases and send numeric limits with a monthly budget reset", () => {
    const expected = {
      user_id: "acme",
      alias: "Acme Corp",
      blocked: false,
      max_budget: 25,
      budget_duration: "1mo",
      tpm_limit: 1000,
      rpm_limit: 60,
      fallbacks: null,
    };
    expect(buildCreateCustomerBody(fields)).toEqual(expected);
  });

  test("should leave optional limits empty without allocating model limits", () => {
    const expected = {
      user_id: "acme",
      alias: null,
      blocked: false,
      max_budget: null,
      budget_duration: null,
      tpm_limit: null,
      rpm_limit: null,
      fallbacks: null,
    };
    expect(buildCreateCustomerBody(emptyFields)).toEqual(expected);
  });

  test("should produce no request for a blank customer id", () => {
    expect(buildCreateCustomerBody({ ...fields, userId: "   " })).toBeNull();
  });

  test("should set independent TPM and RPM allowances by model group", () => {
    expect(
      buildCreateCustomerBody({
        ...emptyFields,
        modelLimits: [
          { model: "primary", tpm: "2000", rpm: "0" },
          { model: "backup", tpm: "", rpm: "30" },
        ],
      }),
    ).toMatchObject({
      model_tpm_limit: { primary: 2000 },
      model_rpm_limit: { primary: 0, backup: 30 },
    });
  });
});

describe("buildUpdateCustomerBody", () => {
  test("should preserve the blocked flag without sending budget fields to the customer endpoint", () => {
    const fields = { userId: "acme", alias: "", blocked: true };
    const expected = { user_id: "acme", alias: null, blocked: true, fallbacks: null };
    expect(buildUpdateCustomerBody(fields)).toEqual(expected);
  });
});

describe("buildCustomerBudgetBody", () => {
  test("should clear overall limits and removed model limits with explicit null and empty maps", () => {
    const fields = { budget: "", tpm: "", rpm: "", modelLimits: [] };
    const expected = {
      max_budget: null,
      tpm_limit: null,
      rpm_limit: null,
      model_tpm_limit: {},
      model_rpm_limit: {},
    };
    expect(buildCustomerBudgetBody(fields)).toEqual(expected);
  });

  test("should preserve zero limits and parse a decimal budget without changing its reset period", () => {
    const fields = { budget: "12.50", tpm: "0", rpm: "0" };
    const expected = {
      max_budget: 12.5,
      tpm_limit: 0,
      rpm_limit: 0,
      model_tpm_limit: {},
      model_rpm_limit: {},
    };
    expect(buildCustomerBudgetBody(fields)).toEqual(expected);
  });
});

describe("modelRateLimitFields", () => {
  test("should restore groups with either limit including zero and groups no longer available", () => {
    expect(modelRateLimitFields({ model_tpm_limit: { retired: 0 }, model_rpm_limit: { primary: 10 } })).toEqual([
      { model: "primary", tpm: "", rpm: "10" },
      { model: "retired", tpm: "0", rpm: "" },
    ]);
  });
});

describe("customer fallback rules", () => {
  test("should distinguish inherited fallbacks from explicitly disabled fallbacks", () => {
    expect(buildCustomerFallbacks(null)).toBeNull();
    expect(buildCustomerFallbacks([])).toEqual([]);
    expect(customerFallbackFields(undefined)).toBeNull();
    expect(customerFallbackFields([])).toEqual([]);
  });

  test("should preserve a customer's fallback order when creating and editing", () => {
    const rules = [{ model: "primary", fallbacks: ["backup", "last-resort"] }];
    const expected = [{ primary: ["backup", "last-resort"] }];
    expect(buildCustomerFallbacks(rules)).toEqual(expected);
    expect(customerFallbackFields(expected)).toEqual(rules);
    const createFields = { userId: "acme", alias: "", budget: "", tpm: "", rpm: "", fallbacks: rules };
    const updateFields = { userId: "acme", alias: "", blocked: false, fallbacks: [] };
    expect(buildCreateCustomerBody(createFields)?.fallbacks).toEqual(expected);
    expect(buildUpdateCustomerBody(updateFields).fallbacks).toEqual([]);
  });

  test("should require a primary and a target for each configured fallback chain", () => {
    expect(hasIncompleteCustomerFallbacks([{ model: "primary", fallbacks: [] }])).toBe(true);
    expect(hasIncompleteCustomerFallbacks([{ model: "", fallbacks: ["backup"] }])).toBe(true);
    expect(hasIncompleteCustomerFallbacks([{ model: "primary", fallbacks: ["backup"] }])).toBe(false);
    expect(hasIncompleteCustomerFallbacks(null)).toBe(false);
    expect(hasIncompleteCustomerFallbacks([])).toBe(false);
  });
});

describe("large rate limit input", () => {
  test.each([
    ["100K", 100000],
    ["1.5M", 1500000],
    ["2b", 2000000000],
    ["1.001K", 1001],
    [".5M", 500000],
    ["1,500,000", 1500000],
    ["0", 0],
    ["", null],
  ])("should parse %s as an exact whole-number allowance", (text, value) => {
    expect(parseRateLimit(String(text))).toEqual({ valid: true, value });
  });

  test.each(["1.5", "1,2", "-1", "100Z", "0.0001K", "9,007,199,254,740,992"])(
    "should reject %s rather than silently changing its value",
    (text) => expect(parseRateLimit(text)).toEqual({ valid: false }),
  );

  test("should show the full number and send numeric values for shorthand in every limit field", () => {
    const fields = {
      budget: "",
      tpm: "1.5M",
      rpm: "100K",
      modelLimits: [{ model: "primary", tpm: "2B", rpm: "1,000" }],
    };
    expect(formatRateLimit("1.5M")).toBe("1,500,000");
    expect(formatRateLimit("invalid")).toBe("invalid");
    const expected = {
      tpm_limit: 1500000,
      rpm_limit: 100000,
      model_tpm_limit: { primary: 2000000000 },
      model_rpm_limit: { primary: 1000 },
    };
    const invalidRateFields = { ...fields, rpm: "1.5" };
    const invalidCreateFields = { ...fields, userId: "acme", alias: "", rpm: "1.5" };
    expect(buildCustomerBudgetBody(fields)).toMatchObject(expected);
    expect(hasInvalidCustomerRateLimits(invalidRateFields)).toBe(true);
    expect(buildCreateCustomerBody(invalidCreateFields)).toBeNull();
  });
});
