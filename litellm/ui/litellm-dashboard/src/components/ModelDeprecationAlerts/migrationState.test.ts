import { describe, expect, it } from "vitest";
import { AlertStatus, effectiveAlertStatus } from "./migrationState";
import type { DeprecationAlert } from "./deprecationAlerts";
const alert: DeprecationAlert = {
  id: "model",
  usageModel: "openai/gpt-4-turbo",
  requests: 1000,
  lastUsedDate: "2026-11-04",
  lastDayRequests: 10,
  model: { model_name: "support", deprecation_date: "2026-10-23", days_until_deprecation: -12, status: "deprecated" },
};
const state = {
  model: alert.usageModel,
  deprecation_date: "2026-10-23",
  status: "resolved" as const,
  last_seen_date: "2026-11-04",
  last_seen_requests: 10,
};
describe("effectiveAlertStatus", () => {
  it("should keep a resolved migration closed when only its historic traffic remains", () => {
    expect(effectiveAlertStatus({ ...alert, state })).toBe(AlertStatus.Resolved);
  });
  it("should reopen on additional requests in the same UTC day", () => {
    expect(effectiveAlertStatus({ ...alert, lastDayRequests: 11, state })).toBe(AlertStatus.Open);
  });
  it("should reopen when the model is used on a later day", () => {
    expect(effectiveAlertStatus({ ...alert, lastUsedDate: "2026-11-05", state })).toBe(AlertStatus.Open);
  });
  it("should keep acknowledgement when old requests roll out of the 30-day window", () => {
    expect(effectiveAlertStatus({ ...alert, requests: 900, state })).toBe(AlertStatus.Resolved);
  });
  it("should allow a shared dismissed state and reopen it on fresh traffic", () => {
    const dismissed = { ...state, status: "dismissed" as const };
    expect(effectiveAlertStatus({ ...alert, state: dismissed })).toBe(AlertStatus.Dismissed);
    expect(effectiveAlertStatus({ ...alert, lastDayRequests: 11, state: dismissed })).toBe(AlertStatus.Open);
  });
});
