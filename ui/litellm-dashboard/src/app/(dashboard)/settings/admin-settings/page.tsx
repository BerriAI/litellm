"use client";

import AdminPanel from "@/components/admins";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useState } from "react";
import useTeams from "@/app/(dashboard)/hooks/useTeams";

const AdminSettings = () => {
  const { teams, setTeams } = useTeams();

  const [searchParams, setSearchParams] = useState<URLSearchParams>(() =>
    typeof window === "undefined" ? new URLSearchParams() : new URLSearchParams(window.location.search),
  );
  const { accessToken, userId, premiumUser, showSSOBanner } = useAuthorized();

  return (
    <AdminPanel
      searchParams={searchParams}
      accessToken={accessToken}
      userID={userId}
      setTeams={setTeams}
      showSSOBanner={showSSOBanner}
      premiumUser={premiumUser}
    />
  );
};

export default AdminSettings;
