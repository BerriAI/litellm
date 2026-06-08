"use client";

import PromptsPanel from "@/components/prompts";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function PromptsRoute() {
  const { accessToken, userRole } = useAuthorized();
  return <PromptsPanel accessToken={accessToken} userRole={userRole} />;
}
