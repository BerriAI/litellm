"use client";

import PromptsPanel from "./components";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function Prompts() {
  const { accessToken, userRole } = useAuthorized();
  return <PromptsPanel accessToken={accessToken} userRole={userRole} />;
}
