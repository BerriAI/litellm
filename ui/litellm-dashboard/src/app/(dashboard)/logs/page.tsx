"use client";

import SpendLogsTable from "@/components/view_logs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";

const LogsPage = () => {
  const { accessToken, token, userRole, userId, premiumUser } = useAuthorized();
  const { teams } = useTeams();

  return (
    <SpendLogsTable
      accessToken={accessToken}
      token={token}
      userRole={userRole}
      userID={userId}
      allTeams={teams || []}
      premiumUser={premiumUser}
    />
  );
};

export default LogsPage;
