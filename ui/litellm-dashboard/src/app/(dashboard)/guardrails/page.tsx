"use client";

import GuardrailsPanel from "@/components/guardrails";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const GuardrailsPage = () => {
  const { accessToken, userRole } = useAuthorized();

  return <GuardrailsPanel accessToken={accessToken} userRole={userRole} />;
};

export default GuardrailsPage;
