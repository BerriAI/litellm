"use client";

import Teams from "@/components/Teams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function TeamsPage() {
  const { accessToken, userId, userRole, isViewOnly, premiumUser } = useAuthorized();
  return (
    <Teams
      accessToken={accessToken}
      userID={userId}
      userRole={userRole}
      isViewOnly={isViewOnly}
      premiumUser={premiumUser ?? false}
    />
  );
}
