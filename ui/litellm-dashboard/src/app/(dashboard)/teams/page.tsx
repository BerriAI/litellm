"use client";

import Teams from "@/components/teams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { useEffect, useState } from "react";
import { Organization } from "@/components/networking";
import { fetchOrganizations } from "@/components/organizations";

const TeamsPage = () => {
  const { accessToken, userId, userRole } = useAuthorized();
  const { teams, setTeams } = useTeams();
  const [searchParams, setSearchParams] = useState<URLSearchParams>(() =>
    typeof window === "undefined" ? new URLSearchParams() : new URLSearchParams(window.location.search),
  );
  const [organizations, setOrganizations] = useState<Organization[]>([]);

  useEffect(() => {
    fetchOrganizations(accessToken, setOrganizations).then(() => {});
  }, [accessToken]);

  return (
    <Teams
      teams={teams}
      searchParams={searchParams}
      accessToken={accessToken}
      setTeams={setTeams}
      userID={userId}
      userRole={userRole}
      organizations={organizations}
    />
  );
};

export default TeamsPage;
