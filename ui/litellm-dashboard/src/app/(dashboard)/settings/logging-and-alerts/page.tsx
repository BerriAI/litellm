"use client";

import Settings from "@/components/settings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const LoggingAndAlertsPage = () => {
  const { accessToken, userRole, userId, premiumUser } = useAuthorized();

  return <Settings accessToken={accessToken} userRole={userRole} userID={userId} premiumUser={premiumUser} />;
};

export default LoggingAndAlertsPage;
