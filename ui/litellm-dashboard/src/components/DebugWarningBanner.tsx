"use client";

import React from "react";
import { TriangleAlert } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";

interface DebugWarningBannerProps {
  accessToken: string | null;
}

export const DebugWarningBanner: React.FC<DebugWarningBannerProps> = ({ accessToken }) => {
  const { data: healthData } = useHealthReadinessDetails(accessToken);

  // Only show banner if detailed debug mode is explicitly enabled
  if (!healthData?.is_detailed_debug) {
    return null;
  }

  return (
    <Alert variant="warning" className="rounded-none border-x-0 border-t-0">
      <TriangleAlert className="size-4" aria-hidden />
      <AlertTitle>Performance Warning: Detailed Debug Mode Active</AlertTitle>
      <AlertDescription>
        Detailed debug logging (<code>LITELLM_LOG=DEBUG</code>) is currently enabled. This mode logs extensive
        diagnostic information and will significantly degrade performance. It should only be used for troubleshooting
        and disabled in production environments.
      </AlertDescription>
    </Alert>
  );
};
