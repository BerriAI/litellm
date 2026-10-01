"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { LensView } from "./_components/LensView";

export default function LensPage() {
  const { accessToken, userRole } = useAuthorized();
  if (!accessToken) return null;
  if (!isProxyAdminTierRole(userRole ?? "")) {
    return <p className="p-6 text-sm text-muted-foreground">Lens requires proxy administrator access.</p>;
  }
  return <LensView accessToken={accessToken} readOnly={!isProxyAdminRole(userRole ?? "")} />;
}
