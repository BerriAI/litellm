import type { RuleInput } from "./usageAlertApi";

export const METRIC_MINIMUMS = { spend: 10, requests: 1000, latency: 500, errors: 10, error_rate: 2 };

export const METRIC_REQUEST_MINIMUMS = { spend: 0, requests: 100, latency: 100, errors: 100, error_rate: 500 };

export const DEFAULT_RULE: RuleInput = {
  name: "",
  scope_type: "deployment",
  scope_id: null,
  metric: "spend",
  direction: "increase",
  multiplier: 3,
  baseline_days: 7,
  min_history_days: 3,
  min_requests: METRIC_REQUEST_MINIMUMS.spend,
  min_value: METRIC_MINIMUMS.spend,
  enabled: true,
  notification_channels: [],
  email_recipients: [],
};

export function ruleInputFromForm(
  data: FormData,
  scope: RuleInput["scope_type"],
  metric: RuleInput["metric"],
): RuleInput {
  const text = (name: string) => String(data.get(name) ?? "").trim();
  const channels = (["email", "slack"] as const).filter((channel) => data.has(channel));
  return {
    name: text("name"),
    scope_type: scope,
    scope_id: scope === "deployment" ? null : text("scope_id") || null,
    metric,
    direction: text("direction") === "decrease" ? "decrease" : "increase",
    multiplier: Number(data.get("multiplier")),
    baseline_days: Number(data.get("baseline_days")),
    min_history_days: Number(data.get("min_history_days")),
    min_requests: Number(data.get("min_requests")),
    min_value: Number(data.get("min_value")),
    enabled: data.has("enabled"),
    notification_channels: channels,
    email_recipients: text("email_recipients")
      .split(",")
      .map((email) => email.trim())
      .filter(Boolean),
  };
}
