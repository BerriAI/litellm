"use client";

import NewUsagePage from "@/components/new_usage";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";

const UsagePage = () => {
  const { accessToken, userRole, userId, premiumUser } = useAuthorized();
  const { teams } = useTeams();

  return (
    <NewUsagePage
      accessToken={accessToken}
      userRole={userRole}
      userID={userId}
      teams={teams ?? []}
      premiumUser={premiumUser}
    />
  );
};

export default UsagePage;
