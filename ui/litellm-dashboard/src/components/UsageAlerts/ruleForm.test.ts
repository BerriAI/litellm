import { describe, expect, it } from "vitest";
import { ruleInputFromForm } from "./ruleForm";

describe("rule form payload", () => {
  it("clears an old entity ID when switching to deployment scope and preserves zero floors", () => {
    const data = new FormData();
    const values = {
      name: " Deployment traffic ",
      scope_id: "old-team",
      multiplier: "2",
      baseline_days: "7",
      min_history_days: "3",
      min_requests: "0",
      min_value: "0",
      direction: "decrease",
    };
    Object.entries(values).forEach(([key, value]) => data.set(key, value));
    const input = ruleInputFromForm(data, "deployment", "requests");
    const expected = {
      name: "Deployment traffic",
      scope_id: null,
      min_requests: 0,
      min_value: 0,
      enabled: false,
      direction: "decrease",
    };
    expect(input).toMatchObject(expected);
  });

  it("uses a wildcard for a blank scoped ID and trims multiple notification recipients", () => {
    const data = new FormData();
    data.set("scope_id", "  ");
    data.set("email", "on");
    data.set("slack", "on");
    data.set("enabled", "on");
    data.set("email_recipients", "first@example.com, , second@example.com ");
    const input = ruleInputFromForm(data, "team", "spend");
    expect(input.scope_id).toBeNull();
    expect(input.notification_channels).toEqual(["email", "slack"]);
    expect(input.email_recipients).toEqual(["first@example.com", "second@example.com"]);
    expect(input.enabled).toBe(true);
  });
});
