"use client";

import GuardrailsPanel from "@/components/guardrails";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const GuardrailsPage = () => {
  const { accessToken, userId, userRole } = useAuthorized();

  return <GuardrailsPanel accessToken={accessToken} userRole={userRole} userID={userId} />;
};

export default GuardrailsPage;
