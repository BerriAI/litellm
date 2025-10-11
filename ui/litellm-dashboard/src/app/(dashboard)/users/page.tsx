"use client";

import ViewUserDashboard from "@/components/view_users";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const UsersPage = () => {
  const { accessToken, userRole, userId, token } = useAuthorized();
  const [keys, setKeys] = useState<null | any[]>([]);

  const { teams } = useTeams();
  const queryClient = new QueryClient();

  return (
    <QueryClientProvider client={queryClient}>
      <ViewUserDashboard
        accessToken={accessToken}
        token={token}
        keys={keys}
        userRole={userRole}
        userID={userId}
        teams={teams as any}
        setKeys={setKeys}
      />
    </QueryClientProvider>
  );
};

export default UsersPage;
