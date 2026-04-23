"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { useState } from "react";
import ModelsAndEndpointsView from "@/app/(dashboard)/models-and-endpoints/ModelsAndEndpointsView";

const ModelsAndEndpointsPage = () => {
  const { token, premiumUser } = useAuthorized();
  const [keys, setKeys] = useState<null | any[]>([]);

  const { teams } = useTeams();

  return (
    <ModelsAndEndpointsView
      token={token}
      modelData={{ data: [] }}
      keys={keys}
      setModelData={() => {}}
      premiumUser={premiumUser}
      teams={teams}
    />
  );
};

export default ModelsAndEndpointsPage;
