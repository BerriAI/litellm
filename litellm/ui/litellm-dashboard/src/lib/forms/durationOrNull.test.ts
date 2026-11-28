import { describe, expect, it } from "vitest";

import { durationOrNull } from "./durationOrNull";

describe("durationOrNull", () => {
  it.each([
    ["", null],
    ["   ", null],
    [undefined, null],
    [null, null],
    [30, null],
  ])("sends %j as null so a cleared duration is removed", (input, expected) => {
    expect(durationOrNull(input)).toBe(expected);
  });

  it("keeps a real duration, trimmed", () => {
    expect(durationOrNull(" 30d ")).toBe("30d");
  });
});