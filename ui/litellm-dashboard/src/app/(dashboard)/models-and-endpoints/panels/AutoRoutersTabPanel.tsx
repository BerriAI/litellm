"use client";

import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { internalUserRoles } from "@/utils/roles";
import { autoRouterCreationScope } from "@/utils/modelPermissions";

import { AutoRoutersPanel } from "../components/AutoRouters/AutoRoutersPanel";

/**
 * Owns the permission decision for the Auto-Routers tab so the panel stays a renderer.
 * Auto routers also admit members of teams that enabled their dedicated management grant.
 * Viewer roles reach the list without write affordances.
 */
export default function AutoRoutersTabPanel() {
  const { accessToken, userRole, userId: userID, isViewOnly } = useAuthorized();
  const { data: teams } = useTeams();
  const { data: uiSettings } = useUISettings();

  const isInternalUser = userRole != null && internalUserRoles.includes(userRole);
  const scope = autoRouterCreationScope(
    { userRole, userID, isViewOnly },
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
      isViewOnly={isViewOnly}
      teams={teams ?? null}
      createScope={scope}
    />
  );
}
