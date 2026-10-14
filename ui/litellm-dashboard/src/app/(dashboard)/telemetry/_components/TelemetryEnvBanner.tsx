"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { TelemetrySettings } from "@/app/(dashboard)/hooks/telemetry/useTelemetrySettings";
import { useTelemetrySettings } from "@/app/(dashboard)/hooks/telemetry/useTelemetrySettings";

export function TelemetryEnvNotice({ settings }: { settings: TelemetrySettings }) {
  if (settings.environment_variables.length === 0) return null;
  return (
    <Alert>
      <AlertTitle>
        {settings.vetoed
          ? "Telemetry is turned off by an environment variable"
          : "Telemetry environment variables are set"}
      </AlertTitle>
      <AlertDescription>
        Set on this proxy: {settings.environment_variables.join(", ")}.{" "}
        {settings.set_by_environment
          ? "These override the settings on this page, which are read-only until they are removed."
          : "The groups below can still be changed here."}
      </AlertDescription>
    </Alert>
  );
}

export default function TelemetryEnvBanner() {
  const { data } = useTelemetrySettings();
  if (data === undefined) return null;
  return (
    <div className="px-4 pt-4 empty:hidden">
      <TelemetryEnvNotice settings={data} />
    </div>
  );
}
