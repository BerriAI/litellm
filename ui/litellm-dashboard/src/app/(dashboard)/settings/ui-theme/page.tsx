"use client";

import UIThemeSettings from "@/components/ui_theme_settings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const UIThemePage = () => {
  const { userId, userRole, accessToken } = useAuthorized();

  return <UIThemeSettings userID={userId} userRole={userRole} accessToken={accessToken} />;
};

export default UIThemePage;
