"use client";

import Settings from "@/components/settings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function LoggingAndAlerts() {
  const { accessToken, userRole, userId, premiumUser } = useAuthorized();
  return <Settings userID={userId} userRole={userRole} accessToken={accessToken} premiumUser={premiumUser} />;
}
