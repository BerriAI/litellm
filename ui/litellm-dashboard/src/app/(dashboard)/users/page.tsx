"use client";

import ViewUserDashboard from "@/components/view_users";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { useState } from "react";

const UsersPage = () => {
  const { accessToken, userRole, userId, token } = useAuthorized();
  const [keys, setKeys] = useState<null | any[]>([]);

  const { teams } = useTeams();

  return (
    <ViewUserDashboard
      accessToken={accessToken}
      token={token}
      keys={keys}
      userRole={userRole}
      userID={userId}
      teams={teams as any}
      setKeys={setKeys}
    />
  );
};

export default UsersPage;
