"use client";

import RequestLogsPanel from "@/components/view_logs/RequestLogsPanel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function RequestLogsPage() {
  const { accessToken, token, userRole, userId } = useAuthorized();
  if (!accessToken || !token) {
    return null;
  }
  if (!userRole || !userId) {
    return null;
  }
  return <RequestLogsPanel accessToken={accessToken} token={token} userRole={userRole} userID={userId} isActive />;
}
