"use client";

import Usage from "./_components/usage";
import { DeprecationBanner } from "@/components/DeprecationBanner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function OldUsagePage() {
  const { accessToken, token, userRole, userId: userID, premiumUser } = useAuthorized();
  return (
    <>
      <DeprecationBanner featureName="The old Usage page" />
      <Usage
        accessToken={accessToken}
        token={token}
        userRole={userRole}
        userID={userID}
        keys={null}
        premiumUser={premiumUser}
      />
    </>
  );
}
