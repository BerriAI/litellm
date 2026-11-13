import { describe, expect, it } from "vitest";
import { requiredFederationValue, validateMaskedValueUntouched } from "./federation_field";

describe("federation field validation", () => {
  it("refuses a hidden stored value that was only partly edited", () => {
    const rule = validateMaskedValueUntouched("os.e****");
    expect(rule("os.e****")).toBe(true);
    expect(rule("os.environ/NEW_REF")).toBe(true);
    expect(rule("os.e****_NEW")).toEqual(expect.stringContaining("Replace the whole value"));
  });

  it.each(["", "  ", "\n", undefined, null])("refuses the blank required value %j", (value) => {
    expect(requiredFederationValue(value)).toBe("Required");
  });

  it("accepts a required value with text in it", () => {
    expect(requiredFederationValue(" svc_1 ")).toBe(true);
  });
});
