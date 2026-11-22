import { fetchClient } from "@/lib/http/api";
import { ApiError } from "@/lib/http/client";
import type { QueryClient } from "@tanstack/react-query";
import type { App, AppResource, AppResourceType } from "./appForms";

export const appDefinitionKey = ["app-definition"] as const;
export const appResourcesKey = (type: AppResourceType | null) => ["app-resources", type] as const;
export const organizationAppKey = (appId: string | null, organizationId: string) =>
  ["app-organization", appId, organizationId] as const;

export const invalidateAppBudgets = (queryClient: QueryClient) =>
  Promise.all(
    ["app-resources", "apps", "app", "app-organization", "app-member", "app-groups", "app-group-organizations"].map(
      (key) => queryClient.invalidateQueries({ queryKey: [key] }),
    ),
  );

export const loadOrganizationApp = async (appId: string, organizationId: string) => {
  try {
    const response = await fetchClient.GET("/management/v1/apps/{app_id}/customer_organizations/{customer_org_id}", {
      params: { path: { app_id: appId, customer_org_id: organizationId } },
    });
    return response.data ?? null;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null;
    throw error;
  }
};

export const loadAppDefinition = async () => {
  const response = await fetchClient.GET("/management/v1/apps/definition");
  if (!response.data) throw new Error("Missing app definition response");
  return response.data;
};

export async function loadAppPages<T>(
  request: (query: { offset: number; limit: number }) => Promise<{ data?: T[] }>,
  offset = 0,
): Promise<T[]> {
  const { data } = await request({ offset, limit: 100 });
  const page = data ?? [];
  return page.length === 100 ? [...page, ...(await loadAppPages(request, offset + 100))] : page;
}

export const loadAppResources = (resourceType: AppResourceType, offset = 0): Promise<AppResource[]> =>
  loadAppPages(
    (query) =>
      fetchClient.GET("/management/v1/apps/resources", {
        params: { query: { ...query, resource_type: resourceType } },
      }),
    offset,
  );

export const loadApps = (offset = 0): Promise<App[]> =>
  loadAppPages((query) => fetchClient.GET("/management/v1/apps", { params: { query } }), offset);

export { appSpend } from "./appBudgetPresentation";
