"use client";

import React from "react";
import { TriangleAlert } from "lucide-react";
import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";
import { useAuth } from "@/contexts/AuthContext";
import { isAdminRole } from "@/utils/roles";

export const EnvCredentialLoginWarningBanner: React.FC<{ accessToken: string | null }> = ({ accessToken }) => {
  const { userRole } = useAuth();
  const { data: healthData } = useHealthReadinessDetails(accessToken);

  if (!isAdminRole(userRole) || !healthData?.show_env_credential_login_warning) {
    return null;
  }

  return (
    <div
      role="alert"
      className="flex items-start gap-3 border-b border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive"
    >
      <TriangleAlert className="mt-0.5 size-5 shrink-0" aria-hidden="true" />
      <div>
        <p className="font-semibold">Environment-credential login is enabled</p>
        <p>
          Anyone with <code className="font-mono">UI_USERNAME</code>/<code className="font-mono">UI_PASSWORD</code> (or
          the master key, when <code className="font-mono">UI_PASSWORD</code> is unset) can sign in as a proxy admin
          with a shared static secret. First create a regular admin account with its own password, then set{" "}
          <code className="font-mono">general_settings.disable_env_credential_login: true</code> to turn this login path
          off.
        </p>
      </div>
    </div>
  );
};
