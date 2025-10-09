"use client";

import ModelDashboard from "@/components/templates/model_dashboard";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { useState } from "react";

const ModelsAndEndpointsPage = () => {
  const { token, accessToken, userRole, userId, premiumUser } = useAuthorized();
  const [keys, setKeys] = useState<null | any[]>([]);

  const { teams } = useTeams();

  return (
    <ModelDashboard
      accessToken={accessToken}
      token={token}
      userRole={userRole}
      userID={userId}
      modelData={{ data: [] }}
      keys={keys}
      setModelData={() => {}}
      premiumUser={premiumUser}
      teams={teams}
    />
  );
};

export default ModelsAndEndpointsPage;
