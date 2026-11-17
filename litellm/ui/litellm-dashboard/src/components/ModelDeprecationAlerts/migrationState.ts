import type { ParseKeys } from "i18next";
import type { DeprecationAlert } from "./deprecationAlerts";

export enum AlertStatus {
  Open = "open",
  Resolved = "resolved",
  Dismissed = "dismissed",
}

export const ALERT_STATUS_LABEL_KEYS = {
  [AlertStatus.Open]: "alerts.status.open",
  [AlertStatus.Resolved]: "alerts.status.resolved",
  [AlertStatus.Dismissed]: "alerts.status.dismissed",
} as const satisfies Record<AlertStatus, ParseKeys>;

export function effectiveAlertStatus(alert: DeprecationAlert): AlertStatus {
  const state = alert.state;
  if (!state || state.status === "open") return AlertStatus.Open;
  if (!state.last_seen_date || alert.lastUsedDate > state.last_seen_date) return AlertStatus.Open;
  if (alert.lastUsedDate === state.last_seen_date && alert.lastDayRequests > state.last_seen_requests)
    return AlertStatus.Open;
  if (state.status === "resolved") return AlertStatus.Resolved;
  return AlertStatus.Dismissed;
}
