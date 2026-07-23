"use client";

import AuditLogsPanel from "@/components/view_logs/AuditLogsPanel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function AuditLogsPage() {
  const { accessToken, token, userRole, userId, premiumUser } = useAuthorized();
  return (
    <AuditLogsPanel
      userID={userId}
      userRole={userRole}
      token={token}
      accessToken={accessToken}
      isActive
      premiumUser={premiumUser}
    />
  );
}
