"use client";

import RouterSettings from "@/components/router_settings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function RouterSettingsPage() {
  const { accessToken, userRole, userId } = useAuthorized();
  return <RouterSettings accessToken={accessToken} userRole={userRole} userID={userId} />;
}
