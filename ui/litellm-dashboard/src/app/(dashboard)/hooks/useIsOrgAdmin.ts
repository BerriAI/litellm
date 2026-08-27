"use client";

import { isOrgAdminForAnyOrg, isOrgAdminSessionRole } from "@/utils/roles";

import { useOrganizations } from "./organizations/useOrganizations";
import useAuthorized from "./useAuthorized";

const useIsOrgAdmin = (): boolean => {
  const { userId, userRole } = useAuthorized();
  const { data: organizations } = useOrganizations();
  return isOrgAdminSessionRole(userRole) || isOrgAdminForAnyOrg(organizations, userId);
};

export default useIsOrgAdmin;
