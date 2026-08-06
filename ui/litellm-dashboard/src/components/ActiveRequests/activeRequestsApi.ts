import { fetchClient } from "@/lib/http/api";
import type { components, operations } from "@/lib/http/schema";

export type ActiveRequest = components["schemas"]["ActiveRequestRecord"];
export type ActiveRequestsResponse = components["schemas"]["ActiveRequestsResponse"];
export type ActiveRequestQuery = NonNullable<
  operations["get_active_requests_global_active_requests_get"]["parameters"]["query"]
>;
export type ActiveRequestFilters = Omit<ActiveRequestQuery, "page" | "page_size">;
export type CancelActiveRequestResponse = components["schemas"]["CancelActiveRequestResponse"];

const detailOf = (error: unknown, fallback: string): string => {
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  return typeof detail === "string" ? detail : fallback;
};

export const activeRequestsCall = async (
  query: ActiveRequestQuery,
  signal?: AbortSignal,
): Promise<ActiveRequestsResponse> => {
  const { data, error } = await fetchClient.GET("/global/active_requests", { params: { query }, signal });
  if (!data) throw new Error(detailOf(error, "Failed to load active requests"));
  return data;
};

export const cancelActiveRequestCall = async (registryId: string): Promise<CancelActiveRequestResponse> => {
  const { data, error } = await fetchClient.POST("/global/active_requests/{registry_id}/cancel", {
    params: { path: { registry_id: registryId } },
  });
  if (!data) throw new Error(detailOf(error, "Failed to cancel the request"));
  return data;
};
