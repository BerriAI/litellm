"use client";

import CacheDashboard from "@/components/cache_dashboard";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function CachingRoute() {
  const { accessToken, userId, userRole, token, premiumUser } = useAuthorized();
  return (
    <CacheDashboard
      userID={userId}
      userRole={userRole}
      token={token}
      accessToken={accessToken}
      premiumUser={premiumUser}
    />
  );
}
