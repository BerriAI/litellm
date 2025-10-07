"use client";

import Sidebar2 from "@/app/(dashboard)/components/Sidebar2";
import Sidebar from "@/components/leftnav";
import React from "react";
import useFeatureFlags from "@/hooks/useFeatureFlags";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

interface SidebarProviderProps {
  defaultSelectedKey: string;
  sidebarCollapsed: boolean;
  setPage: (page: string) => void;
}

const SidebarProvider = ({ defaultSelectedKey, sidebarCollapsed, setPage }: SidebarProviderProps) => {
  const { accessToken, userRole } = useAuthorized();
  const { refactoredUIFlag, setRefactoredUIFlag } = useFeatureFlags();

  return refactoredUIFlag ? (
    <Sidebar2 accessToken={accessToken} userRole={userRole} />
  ) : (
    <Sidebar
      accessToken={accessToken}
      setPage={setPage}
      userRole={userRole}
      defaultSelectedKey={defaultSelectedKey}
      collapsed={sidebarCollapsed}
    />
  );
};

export default SidebarProvider;
