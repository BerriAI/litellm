"use client";

import GeneralSettings from "@/components/general_settings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const RouterSettingsPage = () => {
  const { accessToken, userRole, userId } = useAuthorized();

  return <GeneralSettings accessToken={accessToken} userRole={userRole} userID={userId} modelData={{}} />;
};

export default RouterSettingsPage;
