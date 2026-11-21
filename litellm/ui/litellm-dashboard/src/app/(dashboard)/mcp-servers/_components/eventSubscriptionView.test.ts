import { describe, expect, it } from "vitest";
import { argumentsSummary, canSendTestEvent, canTerminate, statusTone } from "./eventSubscriptionView";

describe("eventSubscriptionView", () => {
  it.each([
    ["active", "success"],
    ["pending", "muted"],
    ["suspended", "warning"],
    ["terminated", "destructive"],
  ] as const)("shows %s as %s", (status, tone) => {
    expect(statusTone(status)).toBe(tone);
  });

  it("summarizes arguments as key=value pairs", () => {
    expect(argumentsSummary({ repo: "acme/web", severity: 1, tags: ["a"] })).toBe("repo=acme/web, severity=1, tags=[\"a\"]");
    expect(argumentsSummary({})).toBe("-");
  });

  it.each([
    ["active", true, true],
    ["suspended", false, true],
    ["pending", false, true],
    ["terminated", false, false],
  ] as const)("a %s subscription can test=%s terminate=%s", (status, test, terminate) => {
    expect(canSendTestEvent({ status })).toBe(test);
    expect(canTerminate({ status })).toBe(terminate);
  });
});
