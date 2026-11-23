"use client";

import { hasCapability, type Capability } from "@/utils/capabilities";

import useAuthorized from "./useAuthorized";
import useIsOrgAdmin from "./useIsOrgAdmin";

const useCan = (capability: Capability): boolean => {
  const { userRole } = useAuthorized();
  const isOrgAdmin = useIsOrgAdmin();
  return hasCapability(userRole, capability, isOrgAdmin);
};

export default useCan;
