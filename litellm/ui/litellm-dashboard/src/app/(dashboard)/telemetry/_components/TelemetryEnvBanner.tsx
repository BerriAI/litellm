"use client";

import { useTranslation } from "react-i18next";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { TelemetrySettings } from "@/app/(dashboard)/hooks/telemetry/useTelemetrySettings";
import { useTelemetrySettings } from "@/app/(dashboard)/hooks/telemetry/useTelemetrySettings";

export function TelemetryEnvNotice({ settings }: { settings: TelemetrySettings }) {
  const { t } = useTranslation();
  if (settings.environment_variables.length === 0) return null;
  return (
    <Alert>
      <AlertTitle>
        {settings.vetoed ? t("shell.telemetryEnvNotice.vetoed") : t("shell.telemetryEnvNotice.set")}
      </AlertTitle>
      <AlertDescription>
        {t("shell.telemetryEnvNotice.setOnProxy", { variables: settings.environment_variables.join(", ") })}{" "}
        {settings.set_by_environment ? t("shell.telemetryEnvNotice.readOnly") : t("shell.telemetryEnvNotice.editable")}
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
