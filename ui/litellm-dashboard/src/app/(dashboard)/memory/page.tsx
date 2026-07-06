"use client";

import { MemoryView } from "./components/MemoryView";
import { DeprecationBanner } from "@/components/DeprecationBanner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Memory() {
  const { accessToken, userRole, userId } = useAuthorized();
  return (
    <>
      <DeprecationBanner featureName="Memory" />
      <MemoryView accessToken={accessToken} userID={userId} userRole={userRole} />
    </>
  );
}
