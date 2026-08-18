import { describe, expect, it } from "vitest";
import { formatAge } from "./ActiveRequestColumns";

const NOW_MS = 1_700_000_000_000;
const startedSecondsAgo = (seconds: number) => NOW_MS / 1000 - seconds;

describe("formatAge", () => {
  it("should report a young request in seconds", () => {
    expect(formatAge(startedSecondsAgo(12), NOW_MS)).toBe("12s");
  });

  it("should switch to minutes at the first full minute", () => {
    expect(formatAge(startedSecondsAgo(59), NOW_MS)).toBe("59s");
    expect(formatAge(startedSecondsAgo(60), NOW_MS)).toBe("1m 0s");
    expect(formatAge(startedSecondsAgo(125), NOW_MS)).toBe("2m 5s");
  });

  it("should switch to hours at the first full hour and drop the seconds", () => {
    expect(formatAge(startedSecondsAgo(3599), NOW_MS)).toBe("59m 59s");
    expect(formatAge(startedSecondsAgo(3600), NOW_MS)).toBe("1h 0m");
    expect(formatAge(startedSecondsAgo(7845), NOW_MS)).toBe("2h 10m");
  });

  it("should clamp a clock skew between proxy and browser to zero", () => {
    expect(formatAge(startedSecondsAgo(-30), NOW_MS)).toBe("0s");
  });
});
