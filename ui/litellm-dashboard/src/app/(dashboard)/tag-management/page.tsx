"use client";

import TagManagement from "@/components/tag_management";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function TagManagementRoute() {
  const { accessToken, userId, userRole } = useAuthorized();
  return <TagManagement accessToken={accessToken} userRole={userRole} userID={userId} />;
}
