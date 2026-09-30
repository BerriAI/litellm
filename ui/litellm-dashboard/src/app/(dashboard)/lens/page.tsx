"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import { EngineView } from "./_components/EngineView";

export default function EnginePage() {
  const { accessToken, userRole } = useAuthorized();
  if (!accessToken) return null;
  return <EngineView accessToken={accessToken} readOnly={!isProxyAdminRole(userRole ?? "")} />;
}
