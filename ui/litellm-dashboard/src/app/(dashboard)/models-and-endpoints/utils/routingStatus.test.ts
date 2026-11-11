import { describe, expect, it } from "vitest";

import { routingStatusToBlocked } from "./routingStatus";

describe("routingStatusToBlocked", () => {
  it.each([
    ["active", false],
    ["paused", true],
  ] as const)("maps %s to blocked=%s", (status, expected) => {
    expect(routingStatusToBlocked(status)).toBe(expected);
  });

  it("returns undefined when there is no status filter", () => {
    expect(routingStatusToBlocked(null)).toBeUndefined();
  });
});
