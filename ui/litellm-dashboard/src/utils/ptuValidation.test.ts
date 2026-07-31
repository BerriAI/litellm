import { describe, expect, it } from "vitest";
import {
  PTU_COUNT_FIELD,
  PTU_RATE_FIELD,
  ptuCountRules,
  ptuPairRule,
  ptuRateRules,
  ptuStartRequiredRule,
} from "./ptuValidation";

const validate = (value: unknown) => ptuCountRules[0].validator(null, value);

describe("ptuCountRules", () => {
  it("accepts positive whole numbers and empty values", async () => {
    await expect(validate(5)).resolves.toBeUndefined();
    await expect(validate("15")).resolves.toBeUndefined();
    await expect(validate("")).resolves.toBeUndefined();
    await expect(validate(null)).resolves.toBeUndefined();
    await expect(validate(undefined)).resolves.toBeUndefined();
  });

  it("rejects fractional values that the backend integer contract would refuse", async () => {
    await expect(validate(2.5)).rejects.toThrow("positive whole number");
    await expect(validate("1.25")).rejects.toThrow("positive whole number");
  });

  it("rejects zero and negatives, which the backend rejects as a non-positive ptu_count", async () => {
    await expect(validate(0)).rejects.toThrow("positive whole number");
    await expect(validate(-1)).rejects.toThrow("positive whole number");
    await expect(validate("-3")).rejects.toThrow("positive whole number");
  });

  it("rejects a value that is not a number at all", async () => {
    await expect(validate("abc")).rejects.toThrow("positive whole number");
  });
});

describe("ptuPairRule", () => {
  const rule = (sibling: unknown) => ptuPairRule(PTU_RATE_FIELD)({ getFieldValue: () => sibling });
  const check = (value: unknown, sibling: unknown) => rule(sibling).validator(null, value);

  it("accepts both set and both cleared, the only shapes the backend stores", async () => {
    await expect(check(10, 2.0)).resolves.toBeUndefined();
    await expect(check("", "")).resolves.toBeUndefined();
    await expect(check(null, undefined)).resolves.toBeUndefined();
  });

  it("rejects a half-set pair, which the backend answers with a 400", async () => {
    await expect(check(10, "")).rejects.toThrow("must be set together");
    await expect(check("", 2.0)).rejects.toThrow("must be set together");
    await expect(check(null, 2.0)).rejects.toThrow("must be set together");
  });

  it("reads the sibling by the field name it was given", () => {
    const seen: string[] = [];
    ptuPairRule(PTU_COUNT_FIELD)({
      getFieldValue: (name: string) => {
        seen.push(name);
        return 1;
      },
    }).validator(null, 1);
    expect(seen).toEqual([PTU_COUNT_FIELD]);
  });
});

describe("ptuRateRules", () => {
  const validate = (value: unknown) => ptuRateRules[0].validator(null, value);

  it("rejects a negative rate, which the backend answers with a 400", async () => {
    await expect(validate(-1)).rejects.toThrow("Cost per PTU / Hour must be a non-negative number");
  });

  it("rejects a negative rate typed as a string, which is what an input yields", async () => {
    await expect(validate("-0.5")).rejects.toThrow("Cost per PTU / Hour must be a non-negative number");
  });

  it("allows zero, which the backend accepts", async () => {
    await expect(validate(0)).resolves.toBeUndefined();
  });

  it("allows a fractional rate", async () => {
    await expect(validate(2.5)).resolves.toBeUndefined();
  });

  it("leaves an empty field to the pair rule", async () => {
    await expect(validate("")).resolves.toBeUndefined();
    await expect(validate(null)).resolves.toBeUndefined();
    await expect(validate(undefined)).resolves.toBeUndefined();
  });

  it("rejects a value that is not a number at all", async () => {
    await expect(validate("abc")).rejects.toThrow("Cost per PTU / Hour must be a non-negative number");
  });
});

describe("ptuStartRequiredRule", () => {
  const rule = (count: unknown, start: unknown) =>
    ptuStartRequiredRule(PTU_COUNT_FIELD)({ getFieldValue: () => count }).validator(null, start);

  it("rejects PTU config with no effective start, which the backend answers with a 400", async () => {
    await expect(rule(10, undefined)).rejects.toThrow("PTU Effective From is required when PTU Count is set");
    await expect(rule(10, "")).rejects.toThrow("PTU Effective From is required when PTU Count is set");
  });

  it("allows a start once given", async () => {
    await expect(rule(10, "2026-08-01T00:00:00Z")).resolves.toBeUndefined();
  });

  it("leaves a deployment with no PTU config alone", async () => {
    await expect(rule(undefined, undefined)).resolves.toBeUndefined();
  });
});
