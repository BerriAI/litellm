"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { internalUserRoles } from "@/utils/roles";
import { modelCreationScope } from "@/utils/modelPermissions";
import AddAdeptRouterTab from "@/components/add_model/AddAdeptRouterTab";

/**
 * Owns the permission decision for the ADEPT Routers tab. Creating an ADEPT router is a
 * POST /model/new like Add Model and Auto Router, so it takes the same audience rule:
 * a proxy admin, or a team admin who scopes it to a team.
 */
export default function AddAdeptRouterPanel() {
  const { accessToken, userRole, userId: userID, isViewOnly } = useAuthorized();
  const { data: teams } = useTeams();
  const { data: uiSettings } = useUISettings();
  const queryClient = useQueryClient();

  const isInternalUser = userRole != null && internalUserRoles.includes(userRole);
  const scope = modelCreationScope(
    { userRole, userID, isViewOnly },
    {
      teams: teams ?? null,
      disabledForInternalUsers: isInternalUser && uiSettings?.values?.disable_model_add_for_internal_users === true,
    },
  );

  return (
    <AddAdeptRouterTab
      handleOk={() => queryClient.invalidateQueries({ queryKey: ["models", "list"] })}
      accessToken={accessToken ?? ""}
      userRole={userRole ?? ""}
      userId={userID ?? null}
      createScope={scope}
    />
  );
}
