import { afterEach, describe, expect, it } from "vitest";

import { formatLocalDate } from "./dateUtils";

const originalTimezone = process.env.TZ;

afterEach(() => {
  if (originalTimezone === undefined) {
    delete process.env.TZ;
  } else {
    process.env.TZ = originalTimezone;
  }
});

describe("formatLocalDate", () => {
  it("preserves a local midnight ahead of UTC", () => {
    process.env.TZ = "Europe/Berlin";
    const localMidnight = new Date(2026, 3, 1);

    expect(localMidnight.toISOString().split("T")[0]).toBe("2026-03-31");
    expect(formatLocalDate(localMidnight)).toBe("2026-04-01");
  });
});
