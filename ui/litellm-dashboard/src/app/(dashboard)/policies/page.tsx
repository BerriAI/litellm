"use client";

import PoliciesPanel from "@/components/policies";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Policies() {
  const { accessToken, userRole } = useAuthorized();
  return <PoliciesPanel accessToken={accessToken} userRole={userRole} />;
}
