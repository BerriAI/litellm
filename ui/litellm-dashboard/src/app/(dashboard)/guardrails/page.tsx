"use client";

import GuardrailsPanel from "@/components/guardrails";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const GuardrailsPage = () => {
  const { accessToken } = useAuthorized();

  return <GuardrailsPanel accessToken={accessToken} />;
};

export default GuardrailsPage;
