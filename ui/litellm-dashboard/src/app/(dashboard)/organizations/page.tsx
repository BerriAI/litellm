"use client";

import OrganizationsTable from "@/components/organizations";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function OrganizationsPage() {
  const { accessToken, userRole, premiumUser } = useAuthorized();
  return <OrganizationsTable userRole={userRole ?? ""} accessToken={accessToken} premiumUser={premiumUser ?? false} />;
}
