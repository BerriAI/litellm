"use client";

import SpendLogsTable from "@/components/view_logs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const LogsPage = () => {
  const { accessToken, token, userRole, userId, premiumUser } = useAuthorized();
  const { teams } = useTeams();

  const queryClient = new QueryClient();

  return (
    <QueryClientProvider client={queryClient}>
      <SpendLogsTable
        accessToken={accessToken}
        token={token}
        userRole={userRole}
        userID={userId}
        allTeams={teams || []}
        premiumUser={premiumUser}
      />
    </QueryClientProvider>
  );
};

export default LogsPage;
