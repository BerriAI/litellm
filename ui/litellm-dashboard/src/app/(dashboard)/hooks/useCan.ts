"use client";

import { hasCapability, type Capability } from "@/utils/capabilities";

import useAuthorized from "./useAuthorized";

const useCan = (capability: Capability): boolean => {
  const { userRole } = useAuthorized();
  return hasCapability(userRole, capability);
};

export default useCan;
