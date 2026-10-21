"use client";

import { Page } from "@/components/shared/Page";
import { teamListCall as v2TeamListCall } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { KeyResponse, Team } from "@/components/key_team_helpers/key_list";
import CreateKey, { type CreateKeyPrefillData } from "@/components/organisms/create_key_button";
import { VirtualKeysTable } from "@/components/VirtualKeysPage/VirtualKeysTable";
import { parseAsArrayOf, parseAsBoolean, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useEffect, useMemo, useState } from "react";

const CREATE_KEY_URL_PARAMS = {
  create: parseAsBoolean.withDefault(false),
  owned_by: parseAsStringLiteral(["you", "service_account", "another_user"] as const),
  team_id: parseAsString,
  key_alias: parseAsString,
  models: parseAsArrayOf(parseAsString),
  key_type: parseAsStringLiteral(["default", "llm_api", "management"] as const),
};

export default function ApiKeysDashboard() {
  const { userId: userID, userRole, accessToken, isViewOnly } = useAuthorized();
  const [{ create, owned_by, team_id, key_alias, models, key_type }] = useQueryStates(CREATE_KEY_URL_PARAMS);

  const [teams, setTeams] = useState<Team[] | null>(null);
  const [keys, setKeys] = useState<KeyResponse[] | null>([]);

  const autoOpenCreate = create;
  const prefillData: CreateKeyPrefillData | undefined = useMemo(() => {
    if (!autoOpenCreate) return undefined;

    if ([owned_by, team_id, key_alias, models, key_type].every((value) => value === null)) {
      return undefined;
    }

    const sanitizedModels = models
      ? models
          .slice(0, 100)
          .map((m) => m.trim().slice(0, 256))
          .filter((m) => m.length > 0)
      : undefined;

    return {
      owned_by: owned_by ?? undefined,
      team_id: team_id?.trim() || undefined,
      key_alias: key_alias === null ? undefined : key_alias.trim().slice(0, 256),
      models: sanitizedModels && sanitizedModels.length > 0 ? sanitizedModels : undefined,
      key_type: key_type ?? undefined,
    };
  }, [autoOpenCreate, key_alias, key_type, models, owned_by, team_id]);

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
    <Page className="h-full">
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
    </Page>
  );
}
