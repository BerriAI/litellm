"use client";

import UIThemeSettings from "@/components/ui_theme_settings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function UiThemeRoute() {
  const { accessToken, userId, userRole } = useAuthorized();
  return <UIThemeSettings userID={userId} userRole={userRole} accessToken={accessToken} />;
}
