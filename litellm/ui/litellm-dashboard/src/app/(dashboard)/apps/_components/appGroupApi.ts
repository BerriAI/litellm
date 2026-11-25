import { useQuery } from "@tanstack/react-query";
import { fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { loadAppPages } from "./appApi";

export type AppGroup = components["schemas"]["AppGroup"];
export type AppGroupOrganization = components["schemas"]["AppGroupOrganization"];

export const loadAppGroups = (): Promise<AppGroup[]> =>
  loadAppPages((query) => fetchClient.GET("/management/v1/app_groups", { params: { query } }));

export const useAppGroups = () => useQuery({ queryKey: ["app-groups"], queryFn: () => loadAppGroups() });

export const loadAppGroupOrganizations = (appGroupId: string): Promise<AppGroupOrganization[]> =>
  loadAppPages((query) =>
    fetchClient.GET("/management/v1/app_groups/{app_group_id}/customer_organizations", {
      params: { path: { app_group_id: appGroupId }, query },
    }),
  );
