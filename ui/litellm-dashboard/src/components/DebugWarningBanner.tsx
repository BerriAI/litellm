"use client";

import React from "react";
import { Alert } from "antd";
import { useHealthReadiness } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadiness";

export const DebugWarningBanner: React.FC = () => {
  const { data: healthData } = useHealthReadiness();

  // Only show banner if detailed debug mode is explicitly enabled
  if (!healthData?.is_detailed_debug) {
    return null;
  }

  return (
    <Alert
      message="Performance Warning: Detailed Debug Mode Active"
      description={
        <>
          Detailed debug logging (<code>LITELLM_LOG=DEBUG</code>) is currently
          enabled. This mode logs extensive diagnostic information and will
          significantly degrade performance. It should only be used for
          troubleshooting and disabled in production environments.
        </>
      }
      type="warning"
      showIcon
      banner
      style={{ marginBottom: 0, borderRadius: 0 }}
    />
  );
};
