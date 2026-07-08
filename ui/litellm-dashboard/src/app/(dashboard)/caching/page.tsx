"use client";

import CacheDashboard from "./components/cache_dashboard";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Caching() {
  const { accessToken, userRole, userId, token, premiumUser } = useAuthorized();
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
