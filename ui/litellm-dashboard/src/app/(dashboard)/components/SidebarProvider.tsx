"use client";

import Sidebar from "@/components/leftnav";
import { getUISettings } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useEffect, useState } from "react";

interface SidebarProviderProps {
  setPage: (page: string) => void;
  defaultSelectedKey: string;
  sidebarCollapsed: boolean;
}

const SidebarProvider = ({ setPage, defaultSelectedKey, sidebarCollapsed }: SidebarProviderProps) => {
  const { accessToken } = useAuthorized();
  const [enabledPagesInternalUsers, setEnabledPagesInternalUsers] = useState<string[] | null>(null);
  const [enableProjectsUI, setEnableProjectsUI] = useState<boolean>(false);
  const [disableAgentsForInternalUsers, setDisableAgentsForInternalUsers] = useState<boolean>(false);
  const [allowAgentsForTeamAdmins, setAllowAgentsForTeamAdmins] = useState<boolean>(false);
  const [disableVectorStoresForInternalUsers, setDisableVectorStoresForInternalUsers] = useState<boolean>(false);
  const [allowVectorStoresForTeamAdmins, setAllowVectorStoresForTeamAdmins] = useState<boolean>(false);

  useEffect(() => {
    const fetchUISettings = async () => {
      if (!accessToken) {
        console.log("[SidebarProvider] No access token, skipping UI settings fetch");
        return;
      }

      try {
        console.log("[SidebarProvider] Fetching UI settings from /get/ui_settings");
        const settings = await getUISettings(accessToken);
        console.log("[SidebarProvider] UI settings response:", settings);
        
        // API returns 'values' not 'settings'
        if (settings?.values?.enabled_ui_pages_internal_users !== undefined) {
          console.log("[SidebarProvider] Setting enabled pages:", settings.values.enabled_ui_pages_internal_users);
          setEnabledPagesInternalUsers(settings.values.enabled_ui_pages_internal_users);
        } else {
          console.log("[SidebarProvider] No enabled_ui_pages_internal_users in response (all pages visible by default)");
        }

        if (settings?.values?.enable_projects_ui !== undefined) {
          setEnableProjectsUI(Boolean(settings.values.enable_projects_ui));
        }

        if (settings?.values?.disable_agents_for_internal_users !== undefined) {
          setDisableAgentsForInternalUsers(Boolean(settings.values.disable_agents_for_internal_users));
        }

        if (settings?.values?.allow_agents_for_team_admins !== undefined) {
          setAllowAgentsForTeamAdmins(Boolean(settings.values.allow_agents_for_team_admins));
        }

        if (settings?.values?.disable_vector_stores_for_internal_users !== undefined) {
          setDisableVectorStoresForInternalUsers(Boolean(settings.values.disable_vector_stores_for_internal_users));
        }

        if (settings?.values?.allow_vector_stores_for_team_admins !== undefined) {
          setAllowVectorStoresForTeamAdmins(Boolean(settings.values.allow_vector_stores_for_team_admins));
        }
      } catch (error) {
        console.error("[SidebarProvider] Failed to fetch UI settings:", error);
      }
    };

    fetchUISettings();
  }, [accessToken]);

  return (
    <Sidebar
      setPage={setPage}
      defaultSelectedKey={defaultSelectedKey}
      collapsed={sidebarCollapsed}
      enabledPagesInternalUsers={enabledPagesInternalUsers}
      enableProjectsUI={enableProjectsUI}
      disableAgentsForInternalUsers={disableAgentsForInternalUsers}
      allowAgentsForTeamAdmins={allowAgentsForTeamAdmins}
      disableVectorStoresForInternalUsers={disableVectorStoresForInternalUsers}
      allowVectorStoresForTeamAdmins={allowVectorStoresForTeamAdmins}
    />
  );
};

export default SidebarProvider;
