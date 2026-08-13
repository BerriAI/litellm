"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import CLISessionsPageContent from "@/components/CLISessionsPage/CLISessionsPage";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import { proxyAdminTierRoles } from "@/utils/roles";

export default function CLISessions() {
  const { userRole } = useAuthorized();
  if (!proxyAdminTierRoles.includes(userRole || "")) {
    return <AdminOnlyNotice pageTitle="CLI Sessions" />;
  }
  return <CLISessionsPageContent />;
}
