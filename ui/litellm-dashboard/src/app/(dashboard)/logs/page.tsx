"use client";

import SpendLogsTable from "@/components/view_logs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function LogsRoute() {
  const { accessToken, userId, userRole, token, premiumUser } = useAuthorized();
  return (
    <SpendLogsTable
      userID={userId}
      userRole={userRole}
      token={token}
      accessToken={accessToken}
      premiumUser={premiumUser}
    />
  );
}
