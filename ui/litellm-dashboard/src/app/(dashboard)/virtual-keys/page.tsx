"use client";

import { useState } from "react";
import useKeyList from "@/components/key_team_helpers/key_list";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import UserDashboard from "@/components/user_dashboard";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { Organization } from "@/components/networking";

const VirtualKeysPage = () => {
  const { accessToken, userRole, userId, premiumUser, userEmail } = useAuthorized();
  const { teams, setTeams } = useTeams();
  const [createClicked, setCreateClicked] = useState<boolean>(false);
  const [organizations, setOrganizations] = useState<Organization[]>([]);

  const queryClient = new QueryClient();

  const { keys, isLoading, error, pagination, refresh, setKeys } = useKeyList({
    selectedKeyAlias: null,
    currentOrg: null,
    accessToken: accessToken || "",
    createClicked,
  });

  const addKey = (data: any) => {
    setKeys((prevData) => (prevData ? [...prevData, data] : [data]));
    setCreateClicked(() => !createClicked);
  };

  return (
    <QueryClientProvider client={queryClient}>
      <UserDashboard
        userID={userId}
        userRole={userRole}
        userEmail={userEmail}
        teams={teams}
        keys={keys}
        setUserRole={() => {}}
        setUserEmail={() => {}}
        setTeams={setTeams}
        setKeys={setKeys}
        premiumUser={premiumUser}
        organizations={organizations}
        addKey={addKey}
        createClicked={createClicked}
      />
    </QueryClientProvider>
  );
};

export default VirtualKeysPage;
