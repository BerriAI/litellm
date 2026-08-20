"use client";

import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { internalUserRoles } from "@/utils/roles";
import { modelCreationScope } from "@/utils/modelPermissions";

import { AutoRoutersPanel } from "../components/AutoRouters/AutoRoutersPanel";

/**
 * Owns the permission decision for the Auto-Routers tab so the panel stays a renderer.
 * Creating an auto router is a POST /model/new, the same endpoint Add Model posts to, so it
 * takes the same audience rule: a proxy admin, or a team admin who scopes it to a team.
 * Viewer roles reach the list without write affordances.
 */
export default function AutoRoutersTabPanel() {
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
    <AutoRoutersPanel
      accessToken={accessToken}
      userRole={userRole ?? ""}
      userID={userID ?? null}
      teams={teams ?? null}
      createScope={scope}
    />
  );
}
