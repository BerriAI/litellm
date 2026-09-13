import { describe, expect, it } from "vitest";
import {
  PTU_COUNT_FIELD,
  PTU_RATE_FIELD,
  ptuCountRules,
  ptuNoUsageCostRule,
  ptuPairRule,
  ptuRateRules,
  ptuStartRequiredRule,
  ptuWindowOrderRule,
  PTU_END_FIELD,
  PTU_START_FIELD,
  MAX_PTU_COUNT,
  MAX_COST_PER_PTU_PER_HOUR,
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
    await expect(validate(2.5)).rejects.toThrow("whole number between 1 and");
    await expect(validate("1.25")).rejects.toThrow("whole number between 1 and");
  });

  it("rejects zero and negatives, which the backend rejects as a non-positive ptu_count", async () => {
    await expect(validate(0)).rejects.toThrow("whole number between 1 and");
    await expect(validate(-1)).rejects.toThrow("whole number between 1 and");
    await expect(validate("-3")).rejects.toThrow("whole number between 1 and");
  });

  it("rejects a value that is not a number at all", async () => {
    await expect(validate("abc")).rejects.toThrow("whole number between 1 and");
  });
});

describe("ptuNoUsageCostRule", () => {
  const check = (value: unknown, count: unknown) =>
    ptuNoUsageCostRule(PTU_COUNT_FIELD)({ getFieldValue: () => count }).validator(null, value);

  it("leaves a deployment without PTU config free to carry any price", async () => {
    await expect(check("2.5", "")).resolves.toBeUndefined();
    await expect(check(0.5, null)).resolves.toBeUndefined();
  });

  it("accepts a blank or zero price alongside PTU config, which is what the backend stores", async () => {
    await expect(check("", 15)).resolves.toBeUndefined();
    await expect(check(null, 15)).resolves.toBeUndefined();
    await expect(check(0, 15)).resolves.toBeUndefined();
    await expect(check("0", 15)).resolves.toBeUndefined();
  });

  it("rejects a non-zero price alongside PTU config, which the backend answers with a 400", async () => {
    await expect(check("2.5", 15)).rejects.toThrow("bills by reserved capacity");
    await expect(check(0.000001, 15)).rejects.toThrow("bills by reserved capacity");
  });

  it("reads the count by the field name it was given", () => {
    const seen: string[] = [];
    ptuNoUsageCostRule(PTU_COUNT_FIELD)({
      getFieldValue: (name: string) => {
        seen.push(name);
        return 15;
      },
    }).validator(null, 0);
    expect(seen).toEqual([PTU_COUNT_FIELD]);
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
    await expect(validate(-1)).rejects.toThrow("must be between 0 and");
  });

  it("rejects a negative rate typed as a string, which is what an input yields", async () => {
    await expect(validate("-0.5")).rejects.toThrow("must be between 0 and");
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
    await expect(validate("abc")).rejects.toThrow("must be between 0 and");
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

describe("ptuWindowOrderRule", () => {
  const form = (values: Record<string, unknown>) => ({ getFieldValue: (name: string) => values[name] });
  const start = new Date("2026-08-01T00:00:00Z");
  const end = new Date("2026-09-01T00:00:00Z");

  it("accepts an ordered window from either bound", async () => {
    await expect(
      ptuWindowOrderRule(PTU_END_FIELD, "start")(form({ [PTU_END_FIELD]: end })).validator(null, start),
    ).resolves.toBeUndefined();
    await expect(
      ptuWindowOrderRule(PTU_START_FIELD, "end")(form({ [PTU_START_FIELD]: start })).validator(null, end),
    ).resolves.toBeUndefined();
  });

  it("rejects an inverted window from either bound", async () => {
    await expect(
      ptuWindowOrderRule(PTU_END_FIELD, "start")(form({ [PTU_END_FIELD]: start })).validator(null, end),
    ).rejects.toThrow("PTU Effective To must be after PTU Effective From");
    await expect(
      ptuWindowOrderRule(PTU_START_FIELD, "end")(form({ [PTU_START_FIELD]: end })).validator(null, start),
    ).rejects.toThrow("PTU Effective To must be after PTU Effective From");
  });

  it("rejects a zero-length window, which the backend also refuses", async () => {
    await expect(
      ptuWindowOrderRule(PTU_END_FIELD, "start")(form({ [PTU_END_FIELD]: start })).validator(null, start),
    ).rejects.toThrow("must be after");
  });

  it("stays silent while either bound is empty, since the window is optional", async () => {
    await expect(ptuWindowOrderRule(PTU_END_FIELD, "start")(form({})).validator(null, start)).resolves.toBeUndefined();
    await expect(
      ptuWindowOrderRule(PTU_START_FIELD, "end")(form({ [PTU_START_FIELD]: start })).validator(null, ""),
    ).resolves.toBeUndefined();
  });
});

describe("backend maximums are mirrored in the form", () => {
  it("rejects a count above the cap and accepts one at it", async () => {
    await expect(ptuCountRules[0].validator(null, String(MAX_PTU_COUNT + 1))).rejects.toThrow("1,000,000");
    await expect(ptuCountRules[0].validator(null, String(MAX_PTU_COUNT))).resolves.toBeUndefined();
  });

  it("rejects a rate above the cap and accepts one at it", async () => {
    await expect(ptuRateRules[0].validator(null, String(MAX_COST_PER_PTU_PER_HOUR + 1))).rejects.toThrow("1,000,000");
    await expect(ptuRateRules[0].validator(null, String(MAX_COST_PER_PTU_PER_HOUR))).resolves.toBeUndefined();
  });

  it("still accepts a zero rate, which the backend allows", async () => {
    await expect(ptuRateRules[0].validator(null, "0")).resolves.toBeUndefined();
  });
});
