"use client";

import React from "react";
import { TriangleAlert } from "lucide-react";
import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";

const BANNER_CONTENT = {
  example_key: {
    title: "The master key is the docs example key sk-1234",
    body: (
      <>
        Anyone who has read the LiteLLM docs can administer this gateway. Generate a strong random key, set it as{" "}
        <code className="font-mono">LITELLM_MASTER_KEY</code> (or{" "}
        <code className="font-mono">general_settings.master_key</code>), and restart the proxy.
      </>
    ),
  },
  missing: {
    title: "No master key is set",
    body: (
      <>
        Every request to this proxy is accepted without authentication, including admin routes. Set{" "}
        <code className="font-mono">LITELLM_MASTER_KEY</code> (or{" "}
        <code className="font-mono">general_settings.master_key</code>) to a strong random key and restart the proxy.
      </>
    ),
  },
} as const;

export const InsecureMasterKeyWarningBanner: React.FC<{ accessToken: string | null }> = ({ accessToken }) => {
  const { data: healthData } = useHealthReadinessDetails(accessToken);
  const reason = healthData?.insecure_master_key_reason;

  if (reason == null) {
    return null;
  }

  const { title, body } = BANNER_CONTENT[reason];

  return (
    <div
      role="alert"
      className="flex items-start gap-3 border-b border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive"
    >
      <TriangleAlert className="mt-0.5 size-5 shrink-0" aria-hidden="true" />
      <div>
        <p className="font-semibold">{title}</p>
        <p>{body}</p>
      </div>
    </div>
  );
};
