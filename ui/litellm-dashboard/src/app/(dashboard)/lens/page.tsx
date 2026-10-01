"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { EngineView } from "./_components/EngineView";

export default function EnginePage() {
  const { accessToken, userRole } = useAuthorized();
  if (!accessToken) return null;
  if (!isProxyAdminTierRole(userRole ?? "")) {
    return <p className="p-6 text-sm text-muted-foreground">Lens requires proxy administrator access.</p>;
  }
  return <EngineView accessToken={accessToken} readOnly={!isProxyAdminRole(userRole ?? "")} />;
}
