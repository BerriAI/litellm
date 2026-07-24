"use client";

import OrganizationsPanel from "./_components/OrganizationsPanel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function OrganizationsPage() {
  const { accessToken, userRole, premiumUser } = useAuthorized();
  return <OrganizationsPanel userRole={userRole ?? ""} accessToken={accessToken} premiumUser={premiumUser ?? false} />;
}
