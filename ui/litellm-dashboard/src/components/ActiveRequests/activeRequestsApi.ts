import { fetchClient } from "@/lib/http/api";
import type { components, operations } from "@/lib/http/schema";

export type ActiveRequest = components["schemas"]["ActiveRequestRecord"];
export type ActiveRequestsResponse = components["schemas"]["ActiveRequestsResponse"];
export type ActiveRequestQuery = NonNullable<
  operations["get_active_requests_global_active_requests_get"]["parameters"]["query"]
>;
export type ActiveRequestFilters = Omit<ActiveRequestQuery, "page" | "page_size">;

export const activeRequestsCall = async (
  query: ActiveRequestQuery,
  signal?: AbortSignal,
): Promise<ActiveRequestsResponse> => {
  const { data } = await fetchClient.GET("/global/active_requests", { params: { query }, signal });
  if (!data) throw new Error("Failed to load active requests");
  return data;
};

export const cancelActiveRequestCall = async (registryId: string): Promise<void> => {
  await fetchClient.POST("/global/active_requests/{registry_id}/cancel", {
    params: { path: { registry_id: registryId } },
  });
};
