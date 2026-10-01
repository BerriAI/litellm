export type RoutingStatusFilter = "active" | "paused" | null;

/**
 * Maps the URL-state status filter to the `blocked` query param of
 * `/v2/model/info`: "active" → false (not blocked), "paused" → true,
 * null → undefined (no filtering).
 */
export const routingStatusToBlocked = (status: RoutingStatusFilter): boolean | undefined =>
  status === null ? undefined : status === "paused";
