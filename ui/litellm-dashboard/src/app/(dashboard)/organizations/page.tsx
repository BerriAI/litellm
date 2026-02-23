"use client";

import Organizations, { fetchOrganizations } from "@/components/organizations";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useEffect, useState } from "react";
import { Organization } from "@/components/networking";
import { fetchUserModels } from "@/components/organisms/create_key_button";

const OrganizationsPage = () => {
  const { userId: userID, accessToken, userRole, premiumUser } = useAuthorized();
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [userModels, setUserModels] = useState<string[]>([]);

  useEffect(() => {
    fetchOrganizations(accessToken, setOrganizations).then(() => {});
  }, [accessToken]);

  useEffect(() => {
    fetchUserModels(userID, userRole, accessToken, setUserModels).then(() => {});
  }, [userID, userRole, accessToken]);

  return (
    <Organizations
      organizations={organizations}
      userRole={userRole}
      userModels={userModels}
      accessToken={accessToken}
      setOrganizations={setOrganizations}
      premiumUser={premiumUser}
    />
  );
};

export default OrganizationsPage;
