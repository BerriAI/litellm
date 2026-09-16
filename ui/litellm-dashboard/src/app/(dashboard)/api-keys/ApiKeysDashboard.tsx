"use client";

import { teamListCall as v2TeamListCall } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { KeyResponse, Team } from "@/components/key_team_helpers/key_list";
import CreateKey from "@/components/organisms/create_key_button";
import { VirtualKeysTable } from "@/components/VirtualKeysPage/VirtualKeysTable";
import { useEffect, useState } from "react";
import { useCreateKeyDeepLink } from "./useCreateKeyDeepLink";

export default function ApiKeysDashboard() {
  const { userId: userID, userRole, accessToken, isViewOnly } = useAuthorized();

  const [teams, setTeams] = useState<Team[] | null>(null);
  const [keys, setKeys] = useState<KeyResponse[] | null>([]);

  const { autoOpenCreate, prefillData, clearDeepLink } = useCreateKeyDeepLink();

  const addKey = (data: KeyResponse) => {
    setKeys((prevData) => (prevData ? [...prevData, data] : [data]));
  };

  useEffect(() => {
    if (accessToken && userID && userRole) {
      v2TeamListCall(accessToken, 1, 100, {
        userID: userRole !== "Admin" && userRole !== "Admin Viewer" ? userID : null,
      })
        .then((response) => setTeams(response.teams ?? []))
        .catch(console.error);
    }
  }, [accessToken, userID, userRole]);

  return (
    <main className="flex h-full flex-col p-8">
      <VirtualKeysTable
        headerActions={
          isViewOnly ? undefined : (
            <CreateKey
              team={null}
              teams={teams}
              data={keys}
              addKey={addKey}
              autoOpenCreate={autoOpenCreate}
              prefillData={prefillData}
              onAutoOpened={clearDeepLink}
            />
          )
        }
      />
    </main>
  );
}
