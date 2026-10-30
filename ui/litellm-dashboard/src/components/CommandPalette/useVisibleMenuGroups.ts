"use client";

import { useMemo } from "react";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useIsOrgAdmin from "@/app/(dashboard)/hooks/useIsOrgAdmin";
import { isUserTeamAdminForAnyTeam } from "@/utils/roles";
import { menuGroups, visibleMenuGroups, type MenuVisibilityContext } from "@/components/leftnav";

export const useVisibleMenuGroups = () => {
  const { userId, userRole, isViewOnly } = useAuthorized();
  const isOrgAdmin = useIsOrgAdmin();
  const { data: teams } = useTeams({ enabled: false });
  const { data: settings } = useUISettings();
  const values = settings?.values;
  const isTeamAdmin = useMemo(() => isUserTeamAdminForAnyTeam(teams ?? null, userId ?? ""), [teams, userId]);

  return useMemo(() => {
    const context: MenuVisibilityContext = {
      userRole,
      isViewOnly,
      isOrgAdmin,
      isTeamAdmin,
      enabledPagesInternalUsers: values?.enabled_ui_pages_internal_users ?? null,
      enableProjectsUI: Boolean(values?.enable_projects_ui),
      disableAgentsForInternalUsers: Boolean(values?.disable_agents_for_internal_users),
      allowAgentsForTeamAdmins: Boolean(values?.allow_agents_for_team_admins),
      disableVectorStoresForInternalUsers: Boolean(values?.disable_vector_stores_for_internal_users),
      allowVectorStoresForTeamAdmins: Boolean(values?.allow_vector_stores_for_team_admins),
    };
    return visibleMenuGroups(menuGroups, context);
  }, [isOrgAdmin, isTeamAdmin, isViewOnly, userRole, values]);
};
