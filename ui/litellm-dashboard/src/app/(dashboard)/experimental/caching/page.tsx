"use client";

import CacheDashboard from "@/components/cache_dashboard";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const CachingPage = () => {
  const { token, accessToken, userRole, userId, premiumUser } = useAuthorized();

  return (
    <CacheDashboard
      accessToken={accessToken}
      token={token}
      userRole={userRole}
      userID={userId}
      premiumUser={premiumUser}
    />
  );
};

export default CachingPage;
