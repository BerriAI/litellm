"use client";

import GuardrailsPanel from "@/components/guardrails";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function GuardrailsRoute() {
  const { accessToken, userRole } = useAuthorized();
  return <GuardrailsPanel accessToken={accessToken} userRole={userRole} />;
}
