"use client";

import PoliciesPanel from "./_components";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Policies() {
  const { accessToken, userRole } = useAuthorized();
  return <PoliciesPanel accessToken={accessToken} userRole={userRole} />;
}
