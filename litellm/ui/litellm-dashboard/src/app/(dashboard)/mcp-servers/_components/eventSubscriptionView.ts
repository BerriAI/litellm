import type { MCPEventSubscription, MCPEventSubscriptionStatus } from "@/components/mcp_tools/types";

export type StatusTone = "success" | "warning" | "muted" | "destructive";

const STATUS_TONES: Readonly<Record<MCPEventSubscriptionStatus, StatusTone>> = {
  active: "success",
  pending: "muted",
  suspended: "warning",
  terminated: "destructive",
};

export function statusTone(status: MCPEventSubscriptionStatus): StatusTone {
  return STATUS_TONES[status];
}

export function argumentsSummary(args: Readonly<Record<string, unknown>>): string {
  const entries = Object.entries(args);
  if (entries.length === 0) return "-";
  return entries
    .map(([key, value]) => `${key}=${typeof value === "string" ? value : JSON.stringify(value)}`)
    .join(", ");
}

export function canSendTestEvent(subscription: Pick<MCPEventSubscription, "status">): boolean {
  return subscription.status === "active";
}

export function canTerminate(subscription: Pick<MCPEventSubscription, "status">): boolean {
  return subscription.status !== "terminated";
}
