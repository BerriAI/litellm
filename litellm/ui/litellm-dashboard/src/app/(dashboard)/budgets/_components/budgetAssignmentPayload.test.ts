import { describe, expect, it } from "vitest";
import type { BudgetAssignment } from "../../hooks/budgets/budgetProfileTypes";
import { assignmentPayload, assignmentValues, localDateTime } from "./budgetAssignmentPayload";

const assignment: BudgetAssignment = {
  subject_type: "user",
  subject_id: "person",
  budget_id: "standard",
  revision: "original",
  budget_override: { budget_limits: [{ budget_duration: "1mo", max_budget: 20 }] },
  budget_override_expires_at: null,
  effective_budget: {
    budget_limits: [
      { budget_duration: "1mo", max_budget: 20 },
      { budget_duration: "12mo", max_budget: 200 },
    ],
  },
  windows: [],
};
const now = Date.parse("2026-01-01T00:00:00Z");
const future = "2026-06-01T10:30:00Z";

describe("assignmentPayload", () => {
  it("sends only the revision for an unchanged draft", () => {
    expect(assignmentPayload(assignmentValues(assignment), assignment, now)).toEqual({
      ok: true,
      payload: { revision: "original" },
    });
  });
  it("changes the profile without resending personal overrides", () => {
    expect(assignmentPayload({ ...assignmentValues(assignment), profile: "" }, assignment, now)).toEqual({
      ok: true,
      payload: { revision: "original", budget_id: null },
    });
  });
  it("sends an annual increase without resending either cap", () => {
    const values = {
      ...assignmentValues(assignment),
      increase: 15,
      increaseDuration: "12mo" as const,
      increaseExpiry: localDateTime(future),
    };
    expect(assignmentPayload(values, assignment, now)).toEqual({
      ok: true,
      payload: {
        revision: "original",
        budget_override: {
          temp_budget_increase: 15,
          temp_budget_expiry: future.replace("Z", ".000Z"),
          temp_budget_duration: "12mo",
        },
      },
    });
  });
  it("preserves a zero cap and removes a cleared cap from the override", () => {
    const baseline = {
      ...assignment,
      budget_override: {
        budget_limits: [
          { budget_duration: "1mo" as const, max_budget: 20 },
          { budget_duration: "12mo" as const, max_budget: 180 },
        ],
      },
    };
    expect(assignmentPayload({ ...assignmentValues(baseline), monthly: null, annual: 0 }, baseline, now)).toEqual({
      ok: true,
      payload: {
        revision: "original",
        budget_override: { budget_limits: [{ budget_duration: "12mo", max_budget: 0 }] },
      },
    });
  });
  it("does not reactivate or resend an expired increase when only a cap changes", () => {
    const baseline = {
      ...assignment,
      budget_override: {
        ...assignment.budget_override,
        temp_budget_increase: 5,
        temp_budget_expiry: "2025-01-01T00:00:00Z",
        temp_budget_duration: "1mo" as const,
      },
    };
    expect(assignmentPayload({ ...assignmentValues(baseline), monthly: 30 }, baseline, now)).toEqual({
      ok: true,
      payload: {
        revision: "original",
        budget_override: { budget_limits: [{ budget_duration: "1mo", max_budget: 30 }] },
      },
    });
  });
  it("requires a future expiry when an increase changes, even for a zero increase", () => {
    expect(assignmentPayload({ ...assignmentValues(assignment), increase: 0 }, assignment, now)).toEqual({
      ok: false,
      field: "increaseExpiry",
      error: "futureExpiryRequired",
    });
  });
  it("clears all increase fields together without touching caps", () => {
    const baseline = {
      ...assignment,
      budget_override: {
        ...assignment.budget_override,
        temp_budget_increase: 5,
        temp_budget_expiry: future,
        temp_budget_duration: "1mo" as const,
      },
    };
    expect(assignmentPayload({ ...assignmentValues(baseline), increase: null }, baseline, now)).toEqual({
      ok: true,
      payload: {
        revision: "original",
        budget_override: {
          temp_budget_increase: null,
          temp_budget_expiry: null,
          temp_budget_duration: null,
        },
      },
    });
  });
  it("reactivates an expired whole override without resending its contents", () => {
    const baseline = { ...assignment, budget_override_expires_at: "2025-01-01T00:00:00Z" };
    expect(assignmentPayload({ ...assignmentValues(baseline), overrideExpiry: "" }, baseline, now)).toEqual({
      ok: true,
      payload: { revision: "original", budget_override_expires_at: null },
    });
  });
  it("sets whole override expiry independently and interprets input in local time", () => {
    expect(
      assignmentPayload(
        {
          ...assignmentValues(assignment),
          overrideExpiry: localDateTime(future),
        },
        assignment,
        now,
      ),
    ).toEqual({
      ok: true,
      payload: { revision: "original", budget_override_expires_at: future.replace("Z", ".000Z") },
    });
  });
  it("clears both the override and its expiry even if the old increase expired", () => {
    expect(
      assignmentPayload({ ...assignmentValues(assignment), clearOverride: true, increase: 5 }, assignment, now),
    ).toEqual({
      ok: true,
      payload: { revision: "original", budget_override: null, budget_override_expires_at: null },
    });
  });
});
