"use client";

import ModelHubTable from "@/components/AIHub/ModelHubTable";
import PublicModelHub from "@/components/public_model_hub";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { canCreateModels } from "@/utils/modelPermissions";
import { isAdminRole } from "@/utils/roles";

export default function ModelHubTablePage() {
  const { accessToken, userRole, userId: userID, premiumUser, isViewOnly } = useAuthorized();
  if (!isAdminRole(userRole)) {
    return <PublicModelHub accessToken={accessToken} isEmbedded={true} />;
  }
  const canAddModels = canCreateModels(
    { userRole, userID, isViewOnly },
    { teams: null, disabledForInternalUsers: false },
  );
  return (
    <ModelHubTable
      accessToken={accessToken}
      publicPage={false}
      premiumUser={premiumUser}
      userRole={userRole}
      canOpenPlayground={!isViewOnly}
      canAddModels={canAddModels}
    />
  );
}
