"use client";

import TagManagement from "./_components";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function TagManagementPage() {
  const { accessToken, userRole, userId } = useAuthorized();
  return <TagManagement accessToken={accessToken} userRole={userRole} userID={userId} />;
}
