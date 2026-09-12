"use client";

import GeneralSettings from "../router-settings/_components/general_settings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function GeneralSettingsPage() {
  const { accessToken, userRole, userId } = useAuthorized();
  return <GeneralSettings section="general" userID={userId} userRole={userRole} accessToken={accessToken} />;
}
