"use client";

import ModelHubTable from "@/components/AIHub/ModelHubTable";
import PublicModelHub from "@/components/public_model_hub";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isAdminRole } from "@/utils/roles";

export default function ModelHubTablePage() {
  const { accessToken, userRole, premiumUser } = useAuthorized();
  if (!isAdminRole(userRole)) {
    return <PublicModelHub accessToken={accessToken} isEmbedded={true} />;
  }
  return <ModelHubTable accessToken={accessToken} publicPage={false} premiumUser={premiumUser} userRole={userRole} />;
}
