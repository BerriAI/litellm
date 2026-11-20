import { fetchClient } from "@/lib/http/api";
import { ApiError } from "@/lib/http/client";
import type { App, AppResource, AppResourceType } from "./appForms";

export const appDefinitionKey = ["app-definition"] as const;
export const appResourcesKey = (type: AppResourceType | null) => ["app-resources", type] as const;
export const organizationAppKey = (appId: string | null, organizationId: string) =>
  ["app-organization", appId, organizationId] as const;

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

export const loadAppResources = async (resourceType: AppResourceType, offset = 0): Promise<AppResource[]> => {
  const response = await fetchClient.GET("/management/v1/apps/resources", {
    params: { query: { resource_type: resourceType, offset, limit: 100 } },
  });
  const page = response.data ?? [];
  return page.length === 100 ? [...page, ...(await loadAppResources(resourceType, offset + 100))] : page;
};

export const loadApps = async (offset = 0): Promise<App[]> => {
  const result = await fetchClient.GET("/management/v1/apps", { params: { query: { offset, limit: 100 } } });
  const page = result.data ?? [];
  return page.length === 100 ? [...page, ...(await loadApps(offset + 100))] : page;
};

export { appSpend } from "./appBudgetPresentation";
