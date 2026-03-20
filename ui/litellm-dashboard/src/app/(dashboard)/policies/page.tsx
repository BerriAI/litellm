"use client";

import PoliciesPanel from "@/components/policies";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const PoliciesPage = () => {
  const { accessToken, userRole } = useAuthorized();

  return (
    <PoliciesPanel
      accessToken={accessToken}
      userRole={userRole}
    />
  );
};

export default PoliciesPage;
