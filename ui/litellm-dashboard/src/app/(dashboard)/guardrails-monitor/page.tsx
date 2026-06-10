"use client";

import GuardrailsMonitorView from "@/components/GuardrailsMonitor/GuardrailsMonitorView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function GuardrailsMonitorRoute() {
  const { accessToken } = useAuthorized();
  return <GuardrailsMonitorView accessToken={accessToken} />;
}
