"use client";

import GeneralSettings from "@/components/general_settings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useAllProxyModels } from "@/app/(dashboard)/hooks/models/useModels";

export default function RouterSettingsPage() {
  const { accessToken, userRole, userId } = useAuthorized();
  const { data: models } = useAllProxyModels();
  const modelData = { data: (models?.data ?? []).map((m) => ({ model_name: m.id })) };
  return <GeneralSettings userID={userId} userRole={userRole} accessToken={accessToken} modelData={modelData} />;
}
