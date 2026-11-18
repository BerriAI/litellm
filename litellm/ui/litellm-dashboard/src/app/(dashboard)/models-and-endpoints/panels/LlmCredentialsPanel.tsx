"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import CredentialsPanel from "@/components/model_add/CredentialsPanel";
import UserConnectionsPanel from "@/components/model_add/UserConnectionsPanel";
import { all_admin_roles } from "@/utils/roles";

export default function LlmCredentialsPanel() {
  const { userRole } = useAuthorized();
  return (
    <div className="flex flex-col gap-6">
      {all_admin_roles.includes(userRole ?? "") && <CredentialsPanel />}
      <UserConnectionsPanel />
    </div>
  );
}
