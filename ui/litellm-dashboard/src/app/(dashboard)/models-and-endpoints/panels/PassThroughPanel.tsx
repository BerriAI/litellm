"use client";

import PassThroughSettings from "@/components/PassThroughSettings/PassThroughSettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function PassThroughPanel() {
  const { accessToken, userRole, userId: userID } = useAuthorized();
  return <PassThroughSettings accessToken={accessToken} userRole={userRole} userID={userID} />;
}
