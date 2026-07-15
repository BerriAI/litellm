"use client";

import GuardrailsMonitorView from "./_components/GuardrailsMonitorView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function GuardrailsMonitor() {
  const { accessToken } = useAuthorized();
  return <GuardrailsMonitorView accessToken={accessToken} />;
}
