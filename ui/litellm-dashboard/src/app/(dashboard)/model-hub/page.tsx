"use client";

import ModelHubTable from "@/components/AIHub/ModelHubTable";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const ModelHubPage = () => {
  const { accessToken, premiumUser, userRole } = useAuthorized();

  return <ModelHubTable accessToken={accessToken} publicPage={false} premiumUser={premiumUser} userRole={userRole} />;
};

export default ModelHubPage;
