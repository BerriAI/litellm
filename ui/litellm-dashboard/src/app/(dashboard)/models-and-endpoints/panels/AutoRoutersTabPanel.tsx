"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";

import { AutoRoutersPanel } from "../components/AutoRouters/AutoRoutersPanel";

/**
 * Owns the permission decision for the Auto-Routers tab so the panel stays a renderer.
 * Creating or editing an auto router is a POST /model/new or PATCH /model/{id}/update, both
 * proxy-admin gated, so viewer roles read the list without write affordances.
 */
export default function AutoRoutersTabPanel() {
  const { accessToken, userRole } = useAuthorized();

  return (
    <AutoRoutersPanel
      accessToken={accessToken}
      userRole={userRole ?? ""}
      canModify={isProxyAdminRole(userRole ?? "")}
    />
  );
}
