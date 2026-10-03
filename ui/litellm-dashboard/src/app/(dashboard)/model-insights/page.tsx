"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import ModelInsightsView from "./_components/ModelInsightsView";

export default function ModelInsightsPage() {
  const { accessToken, userRole } = useAuthorized();
  return <ModelInsightsView accessToken={accessToken} canEditTaskClassifier={isProxyAdminRole(userRole ?? "")} />;
}
