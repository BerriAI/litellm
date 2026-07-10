"use client";

import GuardrailsPanel from "./_components";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Guardrails() {
  const { accessToken, userRole } = useAuthorized();
  return <GuardrailsPanel accessToken={accessToken} userRole={userRole} />;
}
