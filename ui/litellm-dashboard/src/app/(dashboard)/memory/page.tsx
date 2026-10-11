"use client";

import { MemoryView } from "./_components/MemoryView";
import { DeprecationBanner } from "@/components/DeprecationBanner";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useCan from "@/app/(dashboard)/hooks/useCan";

export default function Memory() {
  const { accessToken, userRole, userId } = useAuthorized();
  const canViewMemory = useCan("viewMemory");

  if (!canViewMemory) {
    return <AdminOnlyNotice pageTitle="Memory" />;
  }

  return (
    <>
      <DeprecationBanner featureName="Memory" />
      <MemoryView accessToken={accessToken} userID={userId} userRole={userRole} />
    </>
  );
}
