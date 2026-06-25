"use client";

import Usage from "@/components/usage";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function OldUsagePage() {
  const { accessToken, token, userRole, userId: userID, premiumUser } = useAuthorized();
  return (
    <Usage
      accessToken={accessToken}
      token={token}
      userRole={userRole}
      userID={userID}
      keys={null}
      premiumUser={premiumUser}
    />
  );
}
