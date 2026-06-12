"use client";

import ToolPoliciesView from "@/components/ToolPoliciesView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function ToolPolicies() {
  const { accessToken, userRole } = useAuthorized();
  return <ToolPoliciesView accessToken={accessToken} userRole={userRole} />;
}
