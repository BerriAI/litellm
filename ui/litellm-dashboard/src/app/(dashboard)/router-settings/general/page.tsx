"use client";

import { GeneralConfigTab } from "@/app/(dashboard)/router-settings/_components/general_settings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function GeneralConfigPage() {
  const { accessToken } = useAuthorized();
  if (!accessToken) {
    return null;
  }
  return <GeneralConfigTab accessToken={accessToken} />;
}
