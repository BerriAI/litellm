"use client";

import Usage from "./_components/usage";
import { DeprecationBanner } from "@/components/DeprecationBanner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useLanguage } from "@/contexts/LanguageContext";

export default function OldUsagePage() {
  const { t } = useLanguage();
  const { accessToken, token, userRole, userId: userID, premiumUser } = useAuthorized();
  return (
    <>
      <DeprecationBanner featureName={t("The old Usage page")} />
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
