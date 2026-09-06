"use client";

import { isOrgAdminForAnyOrg, isOrgAdminSessionRole } from "@/utils/roles";

import { useOrganizations } from "./organizations/useOrganizations";
import useAuthorized from "./useAuthorized";

export interface UseIsOrgAdminOptions {
  enabled?: boolean;
}

const useIsOrgAdmin = (options?: UseIsOrgAdminOptions): boolean => {
  const { userId, userRole } = useAuthorized();
  const enabled = options?.enabled ?? true;
  const { data: organizations } = useOrganizations(undefined, { enabled });
  return isOrgAdminSessionRole(userRole) || (enabled && isOrgAdminForAnyOrg(organizations, userId));
};

export default useIsOrgAdmin;
