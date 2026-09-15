"use client";

import React from "react";
import { TriangleAlert } from "lucide-react";
import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";

const REMEDIATION = (
  <>
    You must set <code className="font-mono">LITELLM_MASTER_KEY</code> (or{" "}
    <code className="font-mono">general_settings.master_key</code>) to a strong random key and restart the proxy before
    you can store or use upstream credentials or manage virtual keys.
  </>
);

const BANNER_BODIES = {
  example_key: (
    <>
      The master key is set to the docs example value sk-1234, which does not count as set. {REMEDIATION}
    </>
  ),
  missing: REMEDIATION,
} as const;

export const InsecureMasterKeyWarningBanner: React.FC<{ accessToken: string | null }> = ({ accessToken }) => {
  const { data: healthData } = useHealthReadinessDetails(accessToken);
  const reason = healthData?.insecure_master_key_reason;

  if (reason == null) {
    return null;
  }

  return (
    <div
      role="alert"
      className="flex items-start gap-3 border-b border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive"
    >
      <TriangleAlert className="mt-0.5 size-5 shrink-0" aria-hidden="true" />
      <div>
        <p className="font-semibold">The master key has not been set</p>
        <p>{BANNER_BODIES[reason]}</p>
        {healthData?.stored_credentials_locked === true && (
          <p>Credentials already stored on this proxy cannot be viewed or used until then.</p>
        )}
      </div>
    </div>
  );
};
