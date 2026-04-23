"use client";

import React from "react";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { AlertTriangle } from "lucide-react";
import { useHealthReadiness } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadiness";

export const DebugWarningBanner: React.FC = () => {
  const { data: healthData } = useHealthReadiness();

  if (!healthData?.is_detailed_debug) return null;

  return (
    <Alert variant="default" className="rounded-none border-0 border-b border-amber-300 bg-amber-50 text-amber-800 dark:bg-amber-950/30 dark:text-amber-200 dark:border-amber-800">
      <AlertTriangle className="h-4 w-4" />
      <AlertTitle>Performance Warning: Detailed Debug Mode Active</AlertTitle>
      <AlertDescription>
        Detailed debug logging (<code>LITELLM_LOG=DEBUG</code>) is currently
        enabled. This mode logs extensive diagnostic information and will
        significantly degrade performance. It should only be used for
        troubleshooting and disabled in production environments.
      </AlertDescription>
    </Alert>
  );
};
