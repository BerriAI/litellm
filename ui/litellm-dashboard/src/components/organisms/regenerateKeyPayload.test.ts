import { describe, expect, it } from "vitest";
import { buildRegenerateKeyPayload, roundToPrecision, type RegenerateKeyFormValues } from "./regenerateKeyPayload";

const values = (overrides: Partial<RegenerateKeyFormValues> = {}): RegenerateKeyFormValues => ({
  key_alias: "my-test-key",
  max_budget: 100,
  tpm_limit: 5000,
  rpm_limit: 500,
  duration: "30d",
  grace_period: "",
  ...overrides,
});

describe("roundToPrecision", () => {
  it.each([
    [42.567, 42.57],
    [1.005, 1.01],
    [2.675, 2.68],
    [0.125, 0.13],
    [1.0049999, 1],
    [7, 7],
    [1000, 1000],
    [0, 0],
    [0.004, 0],
    [0.005, 0.01],
    [1e-7, 0],
  ])("rounds %s to %s", (input, expected) => {
    expect(roundToPrecision(input, 2)).toBe(expected);
  });

  it("rounds the magnitude of a negative away from zero", () => {
    expect(roundToPrecision(-3.456, 2)).toBe(-3.46);
    expect(roundToPrecision(-1.005, 2)).toBe(-1.01);
  });

  it("does not reintroduce binary floating point noise", () => {
    expect(roundToPrecision(1.1 + 2.2, 2)).toBe(3.3);
    expect(String(roundToPrecision(8.165, 2))).toBe("8.17");
  });

  it("honours a precision other than two", () => {
    expect(roundToPrecision(1.2345, 3)).toBe(1.235);
    expect(roundToPrecision(1.5, 0)).toBe(2);
  });

  it("returns unroundable values untouched", () => {
    expect(roundToPrecision(Number.POSITIVE_INFINITY, 2)).toBe(Number.POSITIVE_INFINITY);
    expect(roundToPrecision(Number.NaN, 2)).toBeNaN();
    expect(roundToPrecision(Number.MAX_VALUE, 2)).toBe(Number.MAX_VALUE);
  });
});

describe("buildRegenerateKeyPayload", () => {
  it("rounds max_budget and leaves every other field alone", () => {
    expect(buildRegenerateKeyPayload(values({ max_budget: 42.567, tpm_limit: 12.7, rpm_limit: 9.99 }))).toStrictEqual({
      key_alias: "my-test-key",
      max_budget: 42.57,
      tpm_limit: 12.7,
      rpm_limit: 9.99,
      duration: "30d",
      grace_period: "",
    });
  });

  it.each([
    ["null", null],
    ["undefined", undefined],
  ])("passes a %s max_budget through without coercing it to a number", (_label, budget) => {
    expect(buildRegenerateKeyPayload(values({ max_budget: budget })).max_budget).toBe(budget);
  });

  it("keeps unset fields as undefined so the request body omits them", () => {
    const payload = buildRegenerateKeyPayload(
      values({ key_alias: undefined, max_budget: undefined, tpm_limit: undefined, rpm_limit: undefined }),
    );

    expect(payload).toStrictEqual({
      key_alias: undefined,
      max_budget: undefined,
      tpm_limit: undefined,
      rpm_limit: undefined,
      duration: "30d",
      grace_period: "",
    });
    expect(JSON.stringify(payload)).toBe('{"duration":"30d","grace_period":""}');
  });

  it("preserves the field order the proxy request body is serialised in", () => {
    expect(Object.keys(buildRegenerateKeyPayload(values({ max_budget: 1.239 })))).toStrictEqual([
      "key_alias",
      "max_budget",
      "tpm_limit",
      "rpm_limit",
      "duration",
      "grace_period",
    ]);
  });

  it("does not mutate the values it is handed", () => {
    const original = values({ max_budget: 42.567 });

    buildRegenerateKeyPayload(original);

    expect(original.max_budget).toBe(42.567);
  });
});
