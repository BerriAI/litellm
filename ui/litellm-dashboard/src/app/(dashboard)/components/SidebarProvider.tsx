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

  useEffect(() => {
    const fetchUISettings = async () => {
      if (!accessToken) return;

      try {
        const settings = await getUISettings(accessToken);
        if (settings?.settings?.enabled_ui_pages_internal_users !== undefined) {
          setEnabledPagesInternalUsers(settings.settings.enabled_ui_pages_internal_users);
        }
      } catch (error) {
        console.error("Failed to fetch UI settings:", error);
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
    />
  );
};

export default SidebarProvider;
