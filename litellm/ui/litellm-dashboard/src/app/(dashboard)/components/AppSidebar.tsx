"use client";

import Sidebar from "@/components/leftnav";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";

const AppSidebar = () => {
  const { data: settings } = useUISettings();
  const values = settings?.values;

  return (
    <Sidebar
      enabledPagesInternalUsers={values?.enabled_ui_pages_internal_users ?? null}
      enableProjectsUI={Boolean(values?.enable_projects_ui)}
      disableAgentsForInternalUsers={Boolean(values?.disable_agents_for_internal_users)}
      allowAgentsForTeamAdmins={Boolean(values?.allow_agents_for_team_admins)}
      disableVectorStoresForInternalUsers={Boolean(values?.disable_vector_stores_for_internal_users)}
      allowVectorStoresForTeamAdmins={Boolean(values?.allow_vector_stores_for_team_admins)}
    />
  );
};

export default AppSidebar;
