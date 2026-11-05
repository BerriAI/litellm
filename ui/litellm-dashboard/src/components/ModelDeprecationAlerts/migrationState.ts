import type { DeprecationAlert } from "./deprecationAlerts";

export enum AlertStatus {
  Open = "open",
  Resolved = "resolved",
  Dismissed = "dismissed",
}

export function effectiveAlertStatus(alert: DeprecationAlert): AlertStatus {
  const state = alert.state;
  if (!state || state.status === "open") return AlertStatus.Open;
  if (!state.last_seen_date || alert.lastUsedDate > state.last_seen_date) return AlertStatus.Open;
  if (alert.lastUsedDate === state.last_seen_date && alert.lastDayRequests > state.last_seen_requests)
    return AlertStatus.Open;
  if (state.status === "resolved") return AlertStatus.Resolved;
  return AlertStatus.Dismissed;
}
