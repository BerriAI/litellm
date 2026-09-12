"use client";

import GeneralSettings from "./_components/general_settings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function RouterSettingsPage() {
  const { accessToken, userRole, userId } = useAuthorized();
  return <GeneralSettings section="router" userID={userId} userRole={userRole} accessToken={accessToken} />;
}
