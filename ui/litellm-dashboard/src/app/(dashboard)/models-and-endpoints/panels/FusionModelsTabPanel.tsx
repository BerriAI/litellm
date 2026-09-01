"use client";

import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { internalUserRoles } from "@/utils/roles";
import { modelCreationScope } from "@/utils/modelPermissions";

import { FusionModelsPanel } from "../components/FusionModels/FusionModelsPanel";

export default function FusionModelsTabPanel() {
  const { accessToken, userRole, userId: userID } = useAuthorized();
  const { data: teams } = useTeams();
  const { data: uiSettings } = useUISettings();
  const isInternalUser = userRole != null && internalUserRoles.includes(userRole);
  const scope = modelCreationScope(
    { userRole, userID },
    {
      teams: teams ?? null,
      disabledForInternalUsers: isInternalUser && uiSettings?.values?.disable_model_add_for_internal_users === true,
    },
  );

  return (
    <FusionModelsPanel
      accessToken={accessToken}
      userRole={userRole ?? ""}
      userID={userID ?? null}
      teams={teams ?? null}
      createScope={scope}
    />
  );
}
