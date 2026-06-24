"use client";

import ViewUserDashboard from "@/components/view_users";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";

export default function UsersPage() {
  const { accessToken, token, userRole, userId } = useAuthorized();
  const { data: teams } = useTeams();
  return (
    <ViewUserDashboard
      userID={userId}
      userRole={userRole}
      token={token}
      teams={teams ?? null}
      accessToken={accessToken}
    />
  );
}
