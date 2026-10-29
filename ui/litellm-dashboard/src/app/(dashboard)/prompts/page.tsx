"use client";

import PromptsPanel from "./_components";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Prompts() {
  const { accessToken, userRole } = useAuthorized();
  return <PromptsPanel accessToken={accessToken} userRole={userRole} />;
}
