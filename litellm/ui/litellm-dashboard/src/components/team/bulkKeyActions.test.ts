import { describe, expect, it } from "vitest";
import { buildBulkKeyUpdate } from "./bulkKeyActions";

describe("buildBulkKeyUpdate", () => {
  it("preserves zero budgets and accepts integer rate limits", () => {
    expect(buildBulkKeyUpdate("budget", "0")).toEqual({ max_budget: 0 });
    expect(buildBulkKeyUpdate("budget", "2.5")).toEqual({ max_budget: 2.5 });
    expect(buildBulkKeyUpdate("rpm", "12")).toEqual({ rpm_limit: 12 });
    expect(buildBulkKeyUpdate("tpm", "1000")).toEqual({ tpm_limit: 1000 });
  });
  it("rejects invalid limits instead of sending unintended zero or noninteger values", () => {
    expect(buildBulkKeyUpdate("budget", "")).toBeNull();
    expect(buildBulkKeyUpdate("budget", "-1")).toBeNull();
    expect(buildBulkKeyUpdate("rpm", "1.5")).toBeNull();
    expect(buildBulkKeyUpdate("tpm", "Infinity")).toBeNull();
    expect(buildBulkKeyUpdate("tpm", "9007199254740992")).toBeNull();
  });
  it("accepts a positive relative expiry and rejects ambiguous or invalid durations", () => {
    expect(buildBulkKeyUpdate("expiry", " 14d ")).toEqual({ duration: "14d" });
    expect(buildBulkKeyUpdate("expiry", "0d")).toBeNull();
    expect(buildBulkKeyUpdate("expiry", "-1d")).toBeNull();
    expect(buildBulkKeyUpdate("expiry", "tomorrow")).toBeNull();
  });
});
