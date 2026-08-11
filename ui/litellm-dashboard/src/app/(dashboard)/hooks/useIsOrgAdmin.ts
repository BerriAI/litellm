"use client";

import { isOrgAdminForAnyOrg, isOrgAdminSessionRole } from "@/utils/roles";

import { useOrganizations } from "./organizations/useOrganizations";
import useAuthorized from "./useAuthorized";

export interface UseIsOrgAdminOptions {
  enabled?: boolean;
}

const useIsOrgAdmin = (options?: UseIsOrgAdminOptions): boolean => {
  const { userId, userRole } = useAuthorized();
  const { data: organizations } = useOrganizations(undefined, options);
  if (options?.enabled === false) {
    return isOrgAdminSessionRole(userRole);
  }
  return isOrgAdminSessionRole(userRole) || isOrgAdminForAnyOrg(organizations, userId);
};

export default useIsOrgAdmin;
