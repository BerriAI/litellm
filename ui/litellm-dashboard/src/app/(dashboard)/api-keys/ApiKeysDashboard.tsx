"use client";

import { teamListCall as v2TeamListCall } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { KeyResponse, Team } from "@/components/key_team_helpers/key_list";
import CreateKey, { CreateKeyPrefillData } from "@/components/organisms/create_key_button";
import { VirtualKeysTable } from "@/components/VirtualKeysPage/VirtualKeysTable";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

export default function ApiKeysDashboard() {
  const { userId: userID, userRole, accessToken, isViewOnly } = useAuthorized();
  const searchParams = useSearchParams()!;

  const [teams, setTeams] = useState<Team[] | null>(null);
  const [keys, setKeys] = useState<KeyResponse[] | null>([]);

  const autoOpenCreate = searchParams.get("create") === "true";
  const prefillData: CreateKeyPrefillData | undefined = useMemo(() => {
    if (!autoOpenCreate) return undefined;

    const ownedBy = searchParams.get("owned_by");
    const teamId = searchParams.get("team_id");
    const keyAlias = searchParams.get("key_alias");
    const modelsParam = searchParams.get("models");
    const keyType = searchParams.get("key_type");

    if (!ownedBy && !teamId && !keyAlias && !modelsParam && !keyType) {
      return undefined;
    }

    const validOwnedByValues = ["you", "service_account", "another_user"];
    const validatedOwnedBy =
      ownedBy && validOwnedByValues.includes(ownedBy) ? (ownedBy as CreateKeyPrefillData["owned_by"]) : undefined;

    const validKeyTypes = ["default", "llm_api", "management"];
    const validatedKeyType =
      keyType && validKeyTypes.includes(keyType) ? (keyType as CreateKeyPrefillData["key_type"]) : undefined;

    const sanitizedKeyAlias = keyAlias ? keyAlias.trim().slice(0, 256) : undefined;

    const sanitizedModels = modelsParam
      ? modelsParam
          .split(",")
          .slice(0, 100)
          .map((m) => m.trim().slice(0, 256))
          .filter((m) => m.length > 0)
      : undefined;

    return {
      owned_by: validatedOwnedBy,
      team_id: teamId?.trim() || undefined,
      key_alias: sanitizedKeyAlias,
      models: sanitizedModels && sanitizedModels.length > 0 ? sanitizedModels : undefined,
      key_type: validatedKeyType,
    };
  }, [searchParams, autoOpenCreate]);

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
            />
          )
        }
      />
    </main>
  );
}
