"use client";

import SpendLogsTable from "@/components/view_logs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const LogsPage = () => {
  const { accessToken, token, userRole, userId, premiumUser } = useAuthorized();

  return (
    <SpendLogsTable
      accessToken={accessToken}
      token={token}
      userRole={userRole}
      userID={userId}
      premiumUser={premiumUser}
    />
  );
};

export default LogsPage;
