import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { all_admin_roles } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";

// ── Types ────────────────────────────────────────────────────────────────────

export type ModelAccessGroupBudget = components["schemas"]["AccessGroupBudget"];
export type ModelAccessGroup = components["schemas"]["AccessGroupInfo"];

// ── Query keys (shared across model-access-group hooks) ──────────────────────

export const modelAccessGroupKeys = createQueryKeys("modelAccessGroups");

// ── Fetch function ───────────────────────────────────────────────────────────

const fetchModelAccessGroups = async (): Promise<ModelAccessGroup[]> => {
  const { data } = await fetchClient.GET("/access_group/list");
  return data?.access_groups ?? [];
};

// ── Hook ─────────────────────────────────────────────────────────────────────

/**
 * Model access groups: the free-text labels on a deployment's `model_info.access_groups`,
 * with the shared budget each one carries. Unrelated to the `/v1/access_group` table that
 * the Access Groups page drives.
 */
export const useModelAccessGroups = () => {
  const { accessToken, userRole } = useAuthorized();

  return useQuery<ModelAccessGroup[]>({
    queryKey: modelAccessGroupKeys.list({}),
    queryFn: fetchModelAccessGroups,
    enabled: Boolean(accessToken) && all_admin_roles.includes(userRole || ""),
  });
};
